#!/usr/bin/env python3
"""
ConvNeXt 眼底 DR 五分类 — 有序多分类头（CORAL）：
- 输出维度为 K-1（5 类 → 4 个 logit），损失为各阈值上的 BCEWithLogits。
- 标签约定：0..4 等级递增（与 QWK 有序性一致）。

数据划分与 train.py 相同：train_80 / val_20(验证+选best) / crossval(最终测试)。

默认启用 warmup+cosine 学习率与验证集 QWK 早停；可用 --constant_lr / --early_stop_patience 0 关闭。
其余（强增强、少数类 Mixup、AMP、日志字段）与 train.py 对齐。

可选 --freeze_backbone：冻结 timm 骨干，仅训练 head（CORAL 4-logit 分类头）。
可选 --emd_weight：在 CORAL BCE 上叠加有序 1D EMD（Wasserstein-1）辅助损失；0 表示关闭。
"""
from __future__ import annotations

import argparse
import csv
import math
import copy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from dr_dataset import APTOSDrDataset
from metrics import compute_dr_metrics


class ModelEMA:
    """Track exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            mv = model_state[k].detach()
            if not torch.is_floating_point(v):
                v.copy_(mv)
            else:
                v.mul_(self.decay).add_(mv, alpha=1.0 - self.decay)


def apply_backbone_freeze(model: torch.nn.Module, freeze: bool) -> int:
    """若 freeze=True，仅 `head.*` 可训练（timm 分类头）。返回可训练参数元素个数。"""
    if not freeze:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("head.")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(n)


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
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(0.82, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.1,
                    # hue>0 在 PIL adjust_hue 上可能触发 uint8 溢出（OverflowError）
                    hue=0.0,
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
        ops = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
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
    """CORAL 二值目标：targets[b,j]=1 当且仅当 labels[b] > j，j=0..K-2。"""
    device = labels.device
    j = torch.arange(num_classes - 1, device=device, dtype=torch.long)
    return (labels.unsqueeze(1) > j).float()


def coral_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """targets: (B, K-1)，硬标签 0/1 或 Mixup 后的软标签 [0,1]。"""
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def coral_probs_monotonic(logits: torch.Tensor) -> torch.Tensor:
    """CORAL logits (B, K-1) -> 单调累积链接下的类别概率 (B, K)。"""
    s = torch.sigmoid(logits)
    # 强制 P(y>0) >= P(y>1) >= ...
    s = torch.flip(
        torch.cummax(torch.flip(s, dims=[1]), dim=1).values,
        dims=[1],
    )
    p0 = 1.0 - s[:, 0]
    pk = s[:, :-1] - s[:, 1:]
    plast = s[:, -1]
    probs = torch.cat([p0.unsqueeze(1), pk, plast.unsqueeze(1)], dim=1)
    probs = probs.clamp(min=1e-6)
    return probs / probs.sum(dim=1, keepdim=True)


def emd1d_ordered(p_pred: torch.Tensor, p_target: torch.Tensor) -> torch.Tensor:
    """有序 K 类上的 1D Wasserstein-1（CDF 之 L1 距离），batch 平均。"""
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


def coral_predict(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """将 (B, K-1) logits 转为类别 0..K-1（各阈值 sigmoid>0.5 的个数）。"""
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).sum(dim=1).long()
    return pred.clamp(0, num_classes - 1)


def mixup_minority_batch_coral(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    minority_classes: set[int],
    mixup_prob: float,
    mixup_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """少数类 Mixup；返回图像、CORAL 软目标 (B, K-1)、类别分布 (B, K)。"""
    device = images.device
    b = images.size(0)
    base = labels_to_coral_targets(labels, num_classes)
    base_probs = labels_to_class_probs(labels, num_classes)
    if mixup_prob <= 0 or not minority_classes:
        return images, base, base_probs

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
    return images_out, target_out, probs_out


@torch.no_grad()
def evaluate_ordinal(model, loader, device, num_classes: int):
    model.eval()
    all_pred, all_true = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred = coral_predict(logits, num_classes).cpu().numpy()
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
        pred = coral_predict(logits, num_classes).cpu().numpy()
        all_pred.append(pred)
        all_true.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    metrics = compute_dr_metrics(y_true, y_pred, num_classes=num_classes)
    val_loss = total_loss / max(n_samples, 1)
    return metrics, val_loss


@torch.no_grad()
def collect_predictions(
    model, loader, device, num_classes: int, use_amp: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (y_true, y_pred)，用于混淆矩阵。"""
    model.eval()
    all_pred, all_true = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        pred = coral_predict(logits, num_classes).cpu().numpy()
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
    """行=真实类别，列=预测类别；保存带表头的 CSV。"""
    cm = sk_confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    )
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
    """按 epoch 更新：前 warmup 线性升温，后续 cosine 衰减。"""
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    warmup_epochs = min(warmup_epochs, total_epochs - 1) if total_epochs > 1 else 0

    def lr_lambda(epoch_idx: int):
        # epoch_idx 从 0 开始，对应第 1 个 step 调度
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        denom = max(total_epochs - warmup_epochs - 1, 1)
        progress = float(epoch_idx - warmup_epochs) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        eta_min_factor = float(eta_min) / max(float(optimizer.defaults["lr"]), 1e-12)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)



def build_layerwise_lr_decay_params(
    model,
    base_lr=1e-4,
    weight_decay=0.01,
):
    decay_map = {
        "head": 1.0,
        "stages.3": 0.7,
        "stages.2": 0.5,
        "stages.1": 0.3,
        "stages.0": 0.15,
        "stem": 0.1,
    }

    param_groups = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        scale = 0.1
        for k, v in decay_map.items():
            if k in name:
                scale = v
                break

        wd = 0.0 if p.ndim < 2 else weight_decay

        param_groups.append(
            {
                "params": [p],
                "lr": base_lr * scale,
                "weight_decay": wd,
            }
        )

    return param_groups

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_root",
        type=str,
        default="/root/autodl-tmp/baseline/AOR-DR/data/APTOS2019",
        help="含 nodr/mild_npdr/... 的目录",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/root/autodl-tmp/baseline/AOR-DR/data/splits",
    )
    parser.add_argument(
        "--train_list",
        type=str,
        default="APTOS_train_80.txt",
        help="训练集 list",
    )
    parser.add_argument(
        "--val_list",
        type=str,
        default="APTOS_val_20.txt",
        help="验证集 list（val_20：每 epoch 与按 QWK 保存 best.pt）",
    )
    parser.add_argument(
        "--test_list",
        type=str,
        default="APTOS_crossval.txt",
        help="测试集 list（crossval：训练结束仅评一次）",
    )
    parser.add_argument("--model", type=str, default="convnext_small.fb_in22k_ft_in1k_384")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
        help="分类头 dropout（timm drop_rate），0 表示关闭",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="", help="仅评测时加载权重路径")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--eval_sets",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help=(
            "与 --eval_only 配合：评测哪些划分。"
            "all=训练+验证+测试各评一次；"
            "val=仅 val_list（val_20）；test=仅 test_list（crossval）"
        ),
    )
    parser.add_argument("--out_dir", type=str, default="./outputs_ordinal_warmup_earlystop")
    parser.add_argument(
        "--log_csv",
        type=str,
        default="training_log.csv",
        help="相对 out_dir，记录 train_loss / val_loss / 验证指标",
    )
    parser.add_argument(
        "--no_strong_aug",
        action="store_true",
        help="关闭强增强，仅保留 Resize + RandomHorizontalFlip",
    )
    parser.add_argument(
        "--no_mixup",
        action="store_true",
        help="关闭少数类 Mixup",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.4,
        help="Mixup Beta(α,α) 参数",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=0.5,
        help="标签属于少数类时，对该样本施加 Mixup 的概率",
    )
    parser.add_argument(
        "--mixup_num_rare",
        type=int,
        default=2,
        help="按训练集样本数，取最少的 K 个类别视为少数类并参与 Mixup",
    )
    parser.add_argument(
        "--constant_lr",
        action="store_true",
        help="关闭 warmup+cosine，使用恒定学习率（与旧版行为一致）",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=3,
        help="线性 warmup 的 epoch 数（与 cosine 调度配合）",
    )
    parser.add_argument(
        "--cosine_eta_min",
        type=float,
        default=None,
        help="cosine 最小学习率，默认 max(1e-6, lr*0.01)",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=15,
        help="验证集 QWK 连续若干 epoch 无提升则提前结束；0 关闭早停",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.001,
        help="QWK 需超过历史最佳至少该幅度才算一次有效提升（用于早停计数重置）",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="冻结骨干网络参数，仅训练分类头 head（CORAL）",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9997,
        help="EMA decay；<=0 关闭 EMA（建议 0.999~0.9999）",
    )
    parser.add_argument(
        "--ema_eval",
        action="store_true",
        default=True,
        help="验证/测试使用 EMA 权重（推荐开启以提升泛化稳定性）",
    )
    parser.add_argument(
        "--gaussian_blur_sigma",
        type=float,
        default=0.0,
        help="高斯平滑标准差 sigma；>0 时在预处理中应用 Gaussian Blur（离散高斯卷积）。",
    )
    parser.add_argument(
        "--gaussian_blur_kernel",
        type=int,
        default=5,
        help="高斯平滑核大小（建议奇数；若给偶数会自动 +1）。",
    )
    parser.add_argument(
        "--emd_weight",
        type=float,
        default=0.0,
        help=(
            "有序 1D EMD（Wasserstein-1）辅助损失权重，叠加在 CORAL BCE 上；"
            "0 表示关闭。建议从 0.1~0.5 试起。"
        ),
    )
    args = parser.parse_args()

    if args.eval_only and not args.checkpoint:
        raise SystemExit("--eval_only 需要同时指定 --checkpoint")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits_dir = Path(args.splits_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    num_classes = 5
    num_coral = num_classes - 1
    model = timm.create_model(
        args.model,
        pretrained=not args.eval_only,
        num_classes=num_coral,
        drop_rate=args.dropout_rate,
    )
    model = model.to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)

    n_trainable = apply_backbone_freeze(model, args.freeze_backbone)
    if args.freeze_backbone and n_trainable == 0:
        raise RuntimeError(
            "freeze_backbone 后无可训练参数：请确认 timm 模型含 head.* 参数"
        )

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

    if args.eval_only:
        split_map = {
            "train": ("train (train_80)", args.train_list),
            "val": ("验证 (val_20)", args.val_list),
            "test": ("测试 (crossval)", args.test_list),
        }
        if args.eval_sets == "all":
            order = ["train", "val", "test"]
        else:
            order = [args.eval_sets]
        use_amp = device.type == "cuda"
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key in order:
            name, fname = split_map[key]
            loader = make_loader(fname, eval_tf, shuffle=False)
            m = evaluate_ordinal(model, loader, device, num_classes)
            print(
                f"[eval_only] {name}: QWK={m['qwk']:.4f} Acc={m['acc']:.4f} F1_macro={m['f1_macro']:.4f}"
            )
            yt, yp = collect_predictions(
                model, loader, device, num_classes, use_amp
            )
            print_and_save_confusion_matrix(
                yt, yp, num_classes, out_dir, split_name=key
            )
        return

    train_loader = make_loader(args.train_list, train_tf, shuffle=True)
    valid_loader = make_loader(args.val_list, eval_tf, shuffle=False)
    test_loader = make_loader(args.test_list, eval_tf, shuffle=False)

    train_list_path = splits_dir / args.train_list
    class_counts = count_class_frequencies(train_list_path, num_classes)
    rare_indices = pick_rarest_class_indices(class_counts, args.mixup_num_rare)
    minority_classes = set(rare_indices) if not args.no_mixup else set()
    use_mixup = bool(minority_classes) and args.mixup_prob > 0 and not args.no_mixup
    use_emd = args.emd_weight > 0.0
    print(
        f"[train] head=CORAL (K-1={num_coral} logits) | 训练集类别计数 {class_counts} | "
        f"少数类(参与 Mixup)={sorted(minority_classes) if use_mixup else '关闭'} | "
        f"EMD辅助损失={'开启 weight=' + str(args.emd_weight) if use_emd else '关闭'} | "
        f"GaussianBlur(sigma={args.gaussian_blur_sigma}, kernel={args.gaussian_blur_kernel}) | "
        f"lr={args.lr} cosine={not args.constant_lr} early_stop_patience={args.early_stop_patience} "
        f"dropout={args.dropout_rate} | "
        f"freeze_backbone={args.freeze_backbone}（可训练参数元素≈{n_trainable}）"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        build_layerwise_lr_decay_params(
            model,
            base_lr=args.lr,
            weight_decay=args.weight_decay,
        )
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
            f"[lr_scheduler] warmup+cosine 开启: warmup_epochs={args.warmup_epochs}, "
            f"eta_min={eta_min:.2e}, init_lr={args.lr:.2e}"
        )
    else:
        print(f"[lr_scheduler] 关闭，恒定学习率 init_lr={args.lr:.2e}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    print(f"[ema] {'enabled' if ema is not None else 'disabled'} | ema_eval={args.ema_eval}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / args.log_csv
    log_fields = [
        "epoch",
        "train_loss",
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
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_mixup:
                images, target_coral, target_probs = mixup_minority_batch_coral(
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
                logits = model(images)
                loss = compute_ordinal_loss(
                    logits, target_coral, target_probs, args.emd_weight
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            total_loss += loss.item() * images.size(0)
        if scheduler is not None:
            scheduler.step()

        n = len(train_loader.dataset)
        train_loss = total_loss / max(n, 1)
        eval_model = ema.ema if (args.ema_eval and ema is not None) else model
        valid_m, val_loss = evaluate_ordinal_metrics_and_loss(
            eval_model,
            valid_loader,
            device,
            num_classes,
            use_amp,
            emd_weight=args.emd_weight,
        )
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{args.epochs} lr={lr_now:.2e} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"验证(val_20) QWK={valid_m['qwk']:.4f} Acc={valid_m['acc']:.4f} F1={valid_m['f1_macro']:.4f}"
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
                # crossval=test 指标在训练结束后才会计算，这里不应写入验证集指标
                "crossval_metrics": None,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "head": "coral",
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
    )
    print(
        f"\n[测试 crossval] test_loss={test_loss:.4f} QWK={test_m['qwk']:.4f} "
        f"Acc={test_m['acc']:.4f} F1_macro={test_m['f1_macro']:.4f}"
    )
    # 混淆矩阵（最佳权重在验证集上选出，此处对 val / test 各输出一份）
    yt_val, yp_val = collect_predictions(
        final_eval_model, valid_loader, device, num_classes, use_amp
    )
    print_and_save_confusion_matrix(
        yt_val, yp_val, num_classes, out_dir, split_name="val"
    )
    yt_test, yp_test = collect_predictions(
        final_eval_model, test_loader, device, num_classes, use_amp
    )
    print_and_save_confusion_matrix(
        yt_test, yp_test, num_classes, out_dir, split_name="test"
    )


if __name__ == "__main__":
    main()
