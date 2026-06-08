#!/usr/bin/env python3
"""
ConvNeXt 眼底 DR 五分类 — CORAL + Memory Bank（软检索原型融合）。

在 train_ordinal.py 基础上，对 backbone 提取的图像表征 f 做 Memory Bank：
  1) 五个等级可学习原型 μ₀..μ₄
  2) 软检索得到 f_proto
  3) 融合 [f; f_proto] 后接 CORAL 头（4 logit）

损失：L_CORAL + λ_emd·L_EMD（与 train_ordinal.py 一致）+ λ_mem·L_memory（可选）。
L_memory 可选 cosine（拉近 f 与 μ_y）或 infonce（对 5 个原型做 CE，等价 prototype InfoNCE）。

数据划分、增强、Mixup、warmup+cosine、早停、EMA 与 train_ordinal.py 对齐。

验证/测试解码与 train_ordinal_emd_infer.py 一致：
  logits → coral_probs_monotonic → argmax 或 round(期望等级)。
训练损失仍用 compute_ordinal_loss（与 emd_infer 相同）。
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from dr_dataset import APTOSDrDataset
from metrics import compute_dr_metrics


# ---------------------------------------------------------------------------
# Memory Bank model
# ---------------------------------------------------------------------------


class TimmFeatureEncoder(nn.Module):
    """timm backbone（无分类头）→ L2 归一化特征 f。"""

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.dim = int(getattr(self.backbone, "num_features", 0))
        if self.dim <= 0:
            raise RuntimeError(f"无法从 {model_name} 读取 num_features")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        elif feat.ndim != 2:
            raise RuntimeError(f"unexpected feature ndim={feat.ndim}")
        return F.normalize(feat, dim=1)


class GradePrototypeBank(nn.Module):
    """五等级原型记忆库 + 软检索。"""

    def __init__(self, num_classes: int, dim: int, temperature: float = 0.07):
        super().__init__()
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.prototypes = nn.Parameter(torch.randn(num_classes, dim) * 0.01)

    def forward(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = F.normalize(self.prototypes, dim=1)
        sim = f @ mu.T
        alpha = F.softmax(sim / self.temperature, dim=1)
        f_proto = alpha @ mu
        return f_proto, alpha, mu


class MemoryFusionHead(nn.Module):
    """concat(f, f_proto) → MLP → CORAL logits。"""

    def __init__(self, dim: int, num_coral: int, dropout: float = 0.2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )
        self.coral_head = nn.Linear(dim, num_coral)

    def forward(self, f: torch.Tensor, f_proto: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(torch.cat([f, f_proto], dim=1))
        return self.coral_head(fused)


class DRMemoryOrdinal(nn.Module):
    """Backbone 表征 + Memory Bank 软检索 + CORAL 头。"""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.2,
        memory_temperature: float = 0.07,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_coral = self.num_classes - 1
        self.encoder = TimmFeatureEncoder(model_name, pretrained=pretrained)
        self.memory = GradePrototypeBank(
            num_classes=self.num_classes,
            dim=self.encoder.dim,
            temperature=memory_temperature,
        )
        self.head = MemoryFusionHead(
            dim=self.encoder.dim,
            num_coral=self.num_coral,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, return_details: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.encoder(x)
        f_proto, alpha, mu = self.memory(f)
        logits = self.head(f, f_proto)
        if return_details:
            return logits, f, alpha, mu
        return logits


# ---------------------------------------------------------------------------
# Shared utilities (from train_ordinal.py)
# ---------------------------------------------------------------------------


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            mv = model_state[k].detach()
            if not torch.is_floating_point(v):
                v.copy_(mv)
            else:
                v.mul_(self.decay).add_(mv, alpha=1.0 - self.decay)


def apply_backbone_freeze(model: DRMemoryOrdinal, freeze: bool) -> int:
    if not freeze:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for name, p in model.named_parameters():
            p.requires_grad = not name.startswith("encoder.backbone.")
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def count_class_frequencies(list_path: Path, num_classes: int) -> List[int]:
    counts = [0] * num_classes
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            label = int(parts[1])
            if 0 <= label < num_classes:
                counts[label] += 1
    return counts


def pick_rarest_class_indices(counts: List[int], k: int) -> List[int]:
    if k <= 0:
        return []
    order = sorted(range(len(counts)), key=lambda i: counts[i])
    return order[: min(k, len(order))]


def build_transforms(
    img_size: int,
    train: bool,
    strong_aug: bool = True,
    gaussian_blur_sigma: float = 0.0,
    gaussian_blur_kernel: int = 5,
):
    if gaussian_blur_kernel % 2 == 0:
        gaussian_blur_kernel += 1
    smooth_op = (
        transforms.GaussianBlur(
            kernel_size=gaussian_blur_kernel,
            sigma=(gaussian_blur_sigma, gaussian_blur_sigma),
        )
        if gaussian_blur_sigma > 0
        else None
    )
    if train:
        if strong_aug:
            ops = [
                transforms.RandomResizedCrop(img_size, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.1, hue=0.0
                ),
            ]
            if smooth_op is not None:
                ops.append(smooth_op)
            ops.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
            return transforms.Compose(ops)
        ops = [transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip()]
        if smooth_op is not None:
            ops.append(smooth_op)
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        return transforms.Compose(ops)
    ops = [transforms.Resize((img_size, img_size))]
    if smooth_op is not None:
        ops.append(smooth_op)
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def labels_to_coral_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    device = labels.device
    j = torch.arange(num_classes - 1, device=device, dtype=torch.long)
    return (labels.unsqueeze(1) > j).float()


def coral_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def coral_probs_monotonic(logits: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(logits)
    s = torch.flip(torch.cummax(torch.flip(s, dims=[1]), dim=1).values, dims=[1])
    p0 = 1.0 - s[:, 0]
    pk = s[:, :-1] - s[:, 1:]
    plast = s[:, -1]
    probs = torch.cat([p0.unsqueeze(1), pk, plast.unsqueeze(1)], dim=1)
    probs = probs.clamp(min=1e-6)
    return probs / probs.sum(dim=1, keepdim=True)


def emd1d_ordered(p_pred: torch.Tensor, p_target: torch.Tensor) -> torch.Tensor:
    cdf_pred = torch.cumsum(p_pred, dim=1)
    cdf_target = torch.cumsum(p_target, dim=1)
    return torch.mean(torch.sum(torch.abs(cdf_pred - cdf_target), dim=1))


def labels_to_class_probs(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).float()


def compute_ordinal_loss(
    logits: torch.Tensor,
    coral_targets: torch.Tensor,
    class_probs: torch.Tensor,
    emd_weight: float,
) -> torch.Tensor:
    loss = coral_loss(logits, coral_targets)
    if emd_weight > 0.0:
        p_pred = coral_probs_monotonic(logits)
        loss = loss + float(emd_weight) * emd1d_ordered(p_pred, class_probs)
    return loss


def memory_alignment_loss(
    f: torch.Tensor,
    mu: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str = "cosine",
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Memory 对齐损失。
    - cosine: mean(1 - cosine(f, μ_y))，仅拉近本类原型
    - infonce: CE( (f @ μᵀ) / τ, y )，拉近本类并压低其他类原型
    """
    if loss_type == "cosine":
        target_proto = mu[labels.long()]
        return (1.0 - (f * target_proto).sum(dim=1)).mean()
    if loss_type == "infonce":
        logits_mem = (f @ mu.T) / float(temperature)
        return F.cross_entropy(logits_mem, labels.long())
    raise ValueError(f"unknown memory loss_type={loss_type!r}, expected cosine or infonce")


def get_memory_weight(max_weight: float, epoch: int, warmup_epochs: int) -> float:
    if max_weight <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return float(max_weight)
    return float(max_weight) * min(float(epoch) / float(warmup_epochs), 1.0)


def emd_predict(
    logits: torch.Tensor,
    num_classes: int,
    decode_mode: str = "argmax",
) -> torch.Tensor:
    """
    EMD 一致推理：与 train_ordinal_emd_infer.py / compute_ordinal_loss 中 p_pred 同源。

    decode_mode:
      - argmax: 取单调 5 类概率最大类（默认）
      - expectation: round(Σ k * p_k)，有序期望等级
    """
    p_pred = coral_probs_monotonic(logits)
    if decode_mode == "argmax":
        return p_pred.argmax(dim=1)
    if decode_mode == "expectation":
        k = torch.arange(num_classes, device=p_pred.device, dtype=p_pred.dtype)
        return (p_pred * k.unsqueeze(0)).sum(dim=1).round().long().clamp(0, num_classes - 1)
    raise ValueError(f"Unknown decode_mode: {decode_mode!r}")


def mixup_minority_batch_coral(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    minority_classes: set[int],
    mixup_prob: float,
    mixup_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 mixed images、CORAL 目标、类别分布、以及是否被 mix 的 mask。"""
    device = images.device
    b = images.size(0)
    base = labels_to_coral_targets(labels, num_classes)
    base_probs = labels_to_class_probs(labels, num_classes)
    mix_mask = torch.zeros(b, dtype=torch.bool, device=device)
    if mixup_prob <= 0 or not minority_classes:
        return images, base, base_probs, mix_mask

    is_minority = torch.zeros(num_classes, dtype=torch.bool, device=device)
    for c in minority_classes:
        if 0 <= c < num_classes:
            is_minority[c] = True
    minority_mask = is_minority[labels]

    perm = torch.randperm(b, device=device)
    lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample((b,)).to(device)
    rand_apply = torch.rand(b, device=device) < mixup_prob
    mix_mask = minority_mask & rand_apply

    lam_x = lam.view(b, 1, 1, 1)
    mixed_images = lam_x * images + (1.0 - lam_x) * images[perm]

    lam_y = lam.view(b, 1)
    mixed_target = lam_y * base + (1.0 - lam_y) * base[perm]
    mixed_probs = lam_y * base_probs + (1.0 - lam_y) * base_probs[perm]

    mix_mask_4d = mix_mask.view(b, 1, 1, 1)
    images_out = torch.where(mix_mask_4d, mixed_images, images)
    mix_mask_2d = mix_mask.unsqueeze(1).expand(-1, num_classes - 1)
    target_out = torch.where(mix_mask_2d, mixed_target, base)
    mix_mask_k = mix_mask.unsqueeze(1).expand(-1, num_classes)
    probs_out = torch.where(mix_mask_k, mixed_probs, base_probs)
    return images_out, target_out, probs_out, mix_mask


@torch.no_grad()
def init_prototypes_from_loader(
    model: DRMemoryOrdinal,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
) -> None:
    """用训练集各类特征均值初始化原型。"""
    model.eval()
    dim = model.encoder.dim
    sums = [torch.zeros(dim, device=device) for _ in range(num_classes)]
    counts = [0] * num_classes
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            f = model.encoder(images)
        for c in range(num_classes):
            mask = labels == c
            if int(mask.sum()) == 0:
                continue
            sums[c] = sums[c] + f[mask].sum(dim=0)
            counts[c] += int(mask.sum())
    for c in range(num_classes):
        if counts[c] > 0:
            proto = F.normalize(sums[c] / float(counts[c]), dim=0)
            model.memory.prototypes.data[c].copy_(proto)
    model.train()
    print(f"[memory] prototypes initialized from class means: counts={counts}")


@torch.no_grad()
def evaluate_ordinal(
    model, loader, device, num_classes: int, decode_mode: str = "argmax"
):
    model.eval()
    all_pred, all_true = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred = emd_predict(logits, num_classes, decode_mode).cpu().numpy()
        all_pred.append(pred)
        all_true.append(labels.numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    return compute_dr_metrics(y_true, y_pred, num_classes=num_classes)


@torch.no_grad()
def evaluate_ordinal_metrics_and_loss(
    model,
    loader,
    device,
    num_classes: int,
    use_amp: bool,
    emd_weight: float = 0.0,
    decode_mode: str = "argmax",
):
    model.eval()
    all_pred, all_true = [], []
    total_loss = 0.0
    n_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            targets = labels_to_coral_targets(labels_dev, num_classes)
            class_probs = labels_to_class_probs(labels_dev, num_classes)
            loss = compute_ordinal_loss(logits, targets, class_probs, emd_weight)
        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        pred = emd_predict(logits, num_classes, decode_mode).cpu().numpy()
        all_pred.append(pred)
        all_true.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    metrics = compute_dr_metrics(y_true, y_pred, num_classes=num_classes)
    val_loss = total_loss / max(n_samples, 1)
    return metrics, val_loss


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
    num_classes: int,
    use_amp: bool,
    decode_mode: str = "argmax",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_pred, all_true = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        pred = emd_predict(logits, num_classes, decode_mode).cpu().numpy()
        all_pred.append(pred)
        all_true.append(labels.numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    return y_true, y_pred


def print_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    out_dir: Path,
    split_name: str,
) -> np.ndarray:
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print(f"\n混淆矩阵 [{split_name}] (行=真实, 列=预测)")
    print(cm)
    path = out_dir / f"confusion_matrix_{split_name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"pred_{j}" for j in range(num_classes)])
        for i in range(num_classes):
            w.writerow([f"true_{i}"] + [int(cm[i, j]) for j in range(num_classes)])
    print(f"  已保存 {path}")
    return cm


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    eta_min: float,
):
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    warmup_epochs = min(warmup_epochs, total_epochs - 1) if total_epochs > 1 else 0

    def lr_lambda(epoch_idx: int):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        denom = max(total_epochs - warmup_epochs - 1, 1)
        progress = float(epoch_idx - warmup_epochs) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        eta_min_factor = float(eta_min) / max(float(optimizer.defaults["lr"]), 1e-12)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_root",
        type=str,
        default="/root/autodl-tmp/baseline/AOR-DR/data/APTOS2019",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/root/autodl-tmp/baseline/AOR-DR/data/splits",
    )
    parser.add_argument("--train_list", type=str, default="APTOS_train_80.txt")
    parser.add_argument("--val_list", type=str, default="APTOS_val_20.txt")
    parser.add_argument("--test_list", type=str, default="APTOS_crossval.txt")
    parser.add_argument("--model", type=str, default="convnext_small.fb_in22k_ft_in1k_384")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Memory fusion MLP dropout，0 表示关闭",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--eval_sets",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs_ordinal_memory_warmup_earlystop",
    )
    parser.add_argument("--log_csv", type=str, default="training_log.csv")
    parser.add_argument("--no_strong_aug", action="store_true")
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--mixup_prob", type=float, default=0.5)
    parser.add_argument("--mixup_num_rare", type=int, default=2)
    parser.add_argument("--constant_lr", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--cosine_eta_min", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="冻结 encoder.backbone，训练 memory + fusion + CORAL 头",
    )
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_eval", action="store_true")
    parser.add_argument("--gaussian_blur_sigma", type=float, default=0.0)
    parser.add_argument("--gaussian_blur_kernel", type=int, default=5)
    parser.add_argument(
        "--emd_weight",
        type=float,
        default=0.0,
        help="有序 1D EMD 辅助损失权重；0 关闭",
    )
    parser.add_argument(
        "--memory_weight",
        type=float,
        default=0.1,
        help="memory 对齐损失权重；0 关闭",
    )
    parser.add_argument(
        "--memory_warmup_epochs",
        type=int,
        default=5,
        help="memory 损失 warmup 轮数；0 表示无 warmup",
    )
    parser.add_argument(
        "--memory_temperature",
        type=float,
        default=0.07,
        help="软检索 softmax 温度；infonce 模式下也用于 memory 损失",
    )
    parser.add_argument(
        "--memory_loss",
        type=str,
        default="cosine",
        choices=["cosine", "infonce"],
        help="memory 对齐损失：cosine=1-cos(f,μ_y)；infonce=对 5 个原型的 CE/InfoNCE",
    )
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="argmax",
        choices=["argmax", "expectation"],
        help="EMD 单调 5 类概率的解码方式：argmax 或 round(期望等级)",
    )
    parser.add_argument(
        "--init_prototypes",
        action="store_true",
        help="训练前用训练集各类特征均值初始化原型",
    )
    args = parser.parse_args()

    if args.eval_only and not args.checkpoint:
        raise SystemExit("--eval_only 需要同时指定 --checkpoint")
    if args.memory_temperature <= 0:
        raise SystemExit("--memory_temperature 必须 > 0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits_dir = Path(args.splits_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = 5
    num_coral = num_classes - 1

    model = DRMemoryOrdinal(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=not args.eval_only,
        dropout=args.dropout_rate,
        memory_temperature=args.memory_temperature,
    ).to(device)
    print(
        f"[inference] EMD-consistent decode: coral_probs_monotonic → {args.decode_mode} "
        f"(训练损失仍用 compute_ordinal_loss)"
    )

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)

    n_trainable = apply_backbone_freeze(model, args.freeze_backbone)
    if args.freeze_backbone and n_trainable == 0:
        raise RuntimeError("freeze_backbone 后无可训练参数")

    train_tf = build_transforms(
        args.img_size,
        train=True,
        strong_aug=not args.no_strong_aug,
        gaussian_blur_sigma=args.gaussian_blur_sigma,
        gaussian_blur_kernel=args.gaussian_blur_kernel,
    )
    eval_tf = build_transforms(
        args.img_size,
        train=False,
        strong_aug=False,
        gaussian_blur_sigma=args.gaussian_blur_sigma,
        gaussian_blur_kernel=args.gaussian_blur_kernel,
    )

    def make_loader(list_name, tf, shuffle):
        ds = APTOSDrDataset(args.images_root, str(splits_dir / list_name), transform=tf)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    use_amp = device.type == "cuda"

    if args.eval_only:
        split_map = {
            "train": ("train (train_80)", args.train_list),
            "val": ("验证 (val_20)", args.val_list),
            "test": ("测试 (crossval)", args.test_list),
        }
        order = ["train", "val", "test"] if args.eval_sets == "all" else [args.eval_sets]
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key in order:
            name, fname = split_map[key]
            loader = make_loader(fname, eval_tf, shuffle=False)
            m = evaluate_ordinal(
                model, loader, device, num_classes, decode_mode=args.decode_mode
            )
            print(
                f"[eval_only] {name}: QWK={m['qwk']:.4f} Acc={m['acc']:.4f} F1_macro={m['f1_macro']:.4f}"
            )
            yt, yp = collect_predictions(
                model,
                loader,
                device,
                num_classes,
                use_amp,
                decode_mode=args.decode_mode,
            )
            print_and_save_confusion_matrix(yt, yp, num_classes, out_dir, split_name=key)
        return

    train_loader = make_loader(args.train_list, train_tf, shuffle=True)
    valid_loader = make_loader(args.val_list, eval_tf, shuffle=False)
    test_loader = make_loader(args.test_list, eval_tf, shuffle=False)

    if args.init_prototypes:
        init_prototypes_from_loader(
            model, train_loader, device, num_classes, use_amp
        )

    train_list_path = splits_dir / args.train_list
    class_counts = count_class_frequencies(train_list_path, num_classes)
    rare_indices = pick_rarest_class_indices(class_counts, args.mixup_num_rare)
    minority_classes = set(rare_indices) if not args.no_mixup else set()
    use_mixup = bool(minority_classes) and args.mixup_prob > 0 and not args.no_mixup
    use_emd = args.emd_weight > 0.0
    use_memory = args.memory_weight > 0.0
    print(
        f"[train] head=CORAL+MemoryBank (K-1={num_coral}) | 训练集类别计数 {class_counts} | "
        f"少数类(参与 Mixup)={sorted(minority_classes) if use_mixup else '关闭'} | "
        f"EMD={'开启 weight=' + str(args.emd_weight) if use_emd else '关闭'} | "
        f"Memory={'开启 weight=' + str(args.memory_weight) if use_memory else '关闭'} "
        f"(loss={args.memory_loss}, warmup={args.memory_warmup_epochs}, "
        f"tau={args.memory_temperature}) | "
        f"GaussianBlur(sigma={args.gaussian_blur_sigma}, kernel={args.gaussian_blur_kernel}) | "
        f"lr={args.lr} cosine={not args.constant_lr} early_stop_patience={args.early_stop_patience} "
        f"dropout={args.dropout_rate} | "
        f"decode_mode={args.decode_mode} | "
        f"freeze_backbone={args.freeze_backbone}（可训练参数元素≈{n_trainable}）"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = None
    if not args.constant_lr:
        eta_min = (
            args.cosine_eta_min
            if args.cosine_eta_min is not None
            else max(1e-6, float(args.lr) * 0.01)
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            eta_min=eta_min,
        )
        print(
            f"[lr_scheduler] warmup+cosine: warmup_epochs={args.warmup_epochs}, "
            f"eta_min={eta_min:.2e}, init_lr={args.lr:.2e}"
        )
    else:
        print(f"[lr_scheduler] 恒定学习率 init_lr={args.lr:.2e}")

    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    print(f"[ema] {'enabled' if ema is not None else 'disabled'} | ema_eval={args.ema_eval}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / args.log_csv
    log_fields = [
        "epoch",
        "train_loss",
        "train_mem_loss",
        "mem_weight",
        "val_loss",
        "val_qwk",
        "val_acc",
        "val_f1_macro",
    ]
    best_qwk = -1.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_mem_loss = 0.0
        mem_weight = get_memory_weight(
            args.memory_weight, epoch, args.memory_warmup_epochs
        )
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mix_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=device)
            if use_mixup:
                images, target_coral, target_probs, mix_mask = mixup_minority_batch_coral(
                    images,
                    labels,
                    num_classes,
                    minority_classes,
                    args.mixup_prob,
                    args.mixup_alpha,
                )
            else:
                target_coral = labels_to_coral_targets(labels, num_classes)
                target_probs = labels_to_class_probs(labels, num_classes)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, f, _alpha, mu = model(images, return_details=True)
                loss = compute_ordinal_loss(
                    logits, target_coral, target_probs, args.emd_weight
                )
                mem_loss = torch.zeros((), device=device)
                if mem_weight > 0:
                    clean_mask = ~mix_mask
                    if int(clean_mask.sum()) > 0:
                        mem_loss = memory_alignment_loss(
                            f[clean_mask],
                            mu,
                            labels[clean_mask],
                            loss_type=args.memory_loss,
                            temperature=args.memory_temperature,
                        )
                        loss = loss + mem_weight * mem_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            total_loss += loss.item() * images.size(0)
            total_mem_loss += mem_loss.item() * images.size(0)

        if scheduler is not None:
            scheduler.step()

        n = len(train_loader.dataset)
        train_loss = total_loss / max(n, 1)
        train_mem_loss = total_mem_loss / max(n, 1)
        eval_model = ema.ema if (args.ema_eval and ema is not None) else model
        valid_m, val_loss = evaluate_ordinal_metrics_and_loss(
            eval_model,
            valid_loader,
            device,
            num_classes,
            use_amp,
            emd_weight=args.emd_weight,
            decode_mode=args.decode_mode,
        )
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{args.epochs} lr={lr_now:.2e} "
            f"train_loss={train_loss:.4f} train_mem_loss={train_mem_loss:.4f} mem_w={mem_weight:.4f} "
            f"val_loss={val_loss:.4f} 验证(val_20) QWK={valid_m['qwk']:.4f} "
            f"Acc={valid_m['acc']:.4f} F1={valid_m['f1_macro']:.4f}"
        )

        write_header = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=log_fields)
            if write_header:
                w.writeheader()
            w.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_mem_loss": train_mem_loss,
                    "mem_weight": mem_weight,
                    "val_loss": val_loss,
                    "val_qwk": valid_m["qwk"],
                    "val_acc": valid_m["acc"],
                    "val_f1_macro": valid_m["f1_macro"],
                }
            )

        improved = valid_m["qwk"] > best_qwk + args.early_stop_min_delta
        if improved:
            best_qwk = valid_m["qwk"]
            ckpt = {
                "model": model.state_dict(),
                "ema_model": (ema.ema.state_dict() if ema is not None else None),
                "epoch": epoch,
                "val_metrics": valid_m,
                "crossval_metrics": None,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "head": "coral_memory",
                "num_classes": num_classes,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best.pt (验证集 val_20 QWK={best_qwk:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                print(
                    f"[early_stop] epoch={epoch}: val QWK 已连续 {epochs_no_improve} 个 epoch "
                    f"未超过历史最佳（min_delta={args.early_stop_min_delta}），提前结束训练。"
                )
                break

    best_path = out_dir / "best.pt"
    if best_path.is_file():
        loaded = torch.load(best_path, map_location=device)
        model.load_state_dict(loaded["model"])
        if args.ema_eval and ema is not None and loaded.get("ema_model") is not None:
            ema.ema.load_state_dict(loaded["ema_model"], strict=True)

    final_eval_model = ema.ema if (args.ema_eval and ema is not None) else model
    test_m, test_loss = evaluate_ordinal_metrics_and_loss(
        final_eval_model,
        test_loader,
        device,
        num_classes,
        use_amp,
        emd_weight=args.emd_weight,
        decode_mode=args.decode_mode,
    )
    print(
        f"\n[测试 crossval] test_loss={test_loss:.4f} QWK={test_m['qwk']:.4f} "
        f"Acc={test_m['acc']:.4f} F1_macro={test_m['f1_macro']:.4f}"
    )
    yt_val, yp_val = collect_predictions(
        final_eval_model,
        valid_loader,
        device,
        num_classes,
        use_amp,
        decode_mode=args.decode_mode,
    )
    print_and_save_confusion_matrix(yt_val, yp_val, num_classes, out_dir, split_name="val")
    yt_test, yp_test = collect_predictions(
        final_eval_model,
        test_loader,
        device,
        num_classes,
        use_amp,
        decode_mode=args.decode_mode,
    )
    print_and_save_confusion_matrix(yt_test, yp_test, num_classes, out_dir, split_name="test")


if __name__ == "__main__":
    main()
