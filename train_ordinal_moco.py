#!/usr/bin/env python3
"""
ConvNeXt 眼底 DR 五分类 — CORAL + MoCo 动量队列（替代 Memory Bank）。

coral_moco 模式：
  1) Encoder_q（可训练）+ Encoder_k（动量副本）双视图
  2) FIFO 特征队列存历史 key 与标签
  3) 检索式融合：f_moco = softmax(f_q @ Q^T / τ_read) @ Q
  4) concat(f_q, f_moco) → CORAL
  5) 监督 MoCo 对比损失 L_moco（正：同图 key + 队列同类；负：队列异类）

损失：L_CORAL + λ_emd·L_EMD + λ_moco·L_moco
Mixup 样本不参与 L_moco（与 Memory 版 L_memory 一致）。

验证/测试前用训练集弱增强视图刷新队列（--refresh_queue_eval）。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dr_dataset import APTOSDrDataset
from metrics import compute_dr_metrics
from train_ordinal_memory import (
    ModelEMA,
    apply_optimizer_lr,
    build_transforms,
    build_warmup_cosine_scheduler,
    collect_predictions,
    compute_training_loss,
    count_class_frequencies,
    evaluate_ordinal_metrics_and_loss,
    lanet_dynamic_lr,
    labels_to_class_probs,
    labels_to_coral_targets,
    mixup_minority_batch_coral,
    mixup_minority_batch_cls,
    pick_rarest_class_indices,
    predict_from_logits,
    print_and_save_confusion_matrix,
)


# ---------------------------------------------------------------------------
# MoCo model
# ---------------------------------------------------------------------------


class TimmFeatureEncoder(nn.Module):
    """timm backbone（无分类头）→ L2 归一化特征。"""

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        import timm

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


class FeatureQueue(nn.Module):
    """FIFO 特征队列（存动量 key 与 DR 标签）。"""

    def __init__(self, dim: int, size: int):
        super().__init__()
        self.size = int(size)
        self.dim = int(dim)
        self.register_buffer("feats", F.normalize(torch.randn(self.size, dim), dim=1))
        self.register_buffer("labels", torch.full((self.size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor, key_labels: torch.Tensor) -> None:
        keys = F.normalize(keys.detach(), dim=1)
        key_labels = key_labels.detach().long()
        n = int(keys.size(0))
        if n <= 0:
            return
        ptr = int(self.ptr[0].item())
        if ptr + n <= self.size:
            self.feats[ptr : ptr + n] = keys
            self.labels[ptr : ptr + n] = key_labels
        else:
            remain = self.size - ptr
            self.feats[ptr:] = keys[:remain]
            self.labels[ptr:] = key_labels[:remain]
            overflow = n - remain
            if overflow > 0:
                self.feats[:overflow] = keys[remain:]
                self.labels[:overflow] = key_labels[remain:]
        self.ptr[0] = (ptr + n) % self.size
        self.count[0] = min(int(self.count[0].item()) + n, self.size)

    def num_valid(self) -> int:
        return int(self.count[0].item())

    def get_valid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.num_valid()
        if n <= 0:
            empty_f = self.feats.new_zeros((0, self.dim))
            empty_l = self.labels.new_zeros((0,), dtype=torch.long)
            return empty_f, empty_l
        return self.feats[:n].clone(), self.labels[:n].clone()

    def read(self, f_q: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n = self.num_valid()
        if n <= 0:
            return torch.zeros_like(f_q), None
        q = F.normalize(self.feats[:n], dim=1)
        sim = (f_q @ q.T) / float(temperature)
        alpha = F.softmax(sim, dim=1)
        f_moco = alpha @ q
        return f_moco, alpha

    @torch.no_grad()
    def reset(self) -> None:
        self.feats.copy_(F.normalize(torch.randn_like(self.feats), dim=1))
        self.labels.fill_(-1)
        self.ptr.zero_()
        self.count.zero_()


class MoCoFusionHead(nn.Module):
    """concat(f_q, f_moco) → MLP → CORAL logits。"""

    def __init__(self, dim: int, num_coral: int, dropout: float = 0.2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )
        self.coral_head = nn.Linear(dim, num_coral)

    def forward(self, f_q: torch.Tensor, f_moco: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(torch.cat([f_q, f_moco], dim=1))
        return self.coral_head(fused)


class CoralHead(nn.Module):
    def __init__(self, dim: int, num_coral: int, dropout: float = 0.2):
        super().__init__()
        drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Sequential(drop, nn.Linear(dim, num_coral))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.head(f)


class ClsHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Sequential(drop, nn.Linear(dim, num_classes))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.head(f)


@torch.no_grad()
def momentum_update_encoder(encoder_q: nn.Module, encoder_k: nn.Module, momentum: float) -> None:
    for pq, pk in zip(encoder_q.parameters(), encoder_k.parameters()):
        pk.data.mul_(float(momentum)).add_(pq.data, alpha=1.0 - float(momentum))


class DRMocoModel(nn.Module):
    """
    head_mode:
      - cls: encoder_q → CE
      - coral: encoder_q → CORAL
      - coral_moco: 双视图 MoCo + 队列检索融合 → CORAL
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.2,
        queue_size: int = 4096,
        read_temperature: float = 0.07,
        head_mode: str = "coral_moco",
    ):
        super().__init__()
        self.head_mode = str(head_mode)
        if self.head_mode not in ("cls", "coral", "coral_moco"):
            raise ValueError(f"head_mode must be cls|coral|coral_moco, got {head_mode!r}")
        self.num_classes = int(num_classes)
        self.num_coral = self.num_classes - 1
        self.read_temperature = float(read_temperature)

        self.encoder_q = TimmFeatureEncoder(model_name, pretrained=pretrained)
        dim = self.encoder_q.dim

        if self.head_mode == "cls":
            self.cls_head = ClsHead(dim, self.num_classes, dropout=dropout)
        elif self.head_mode == "coral":
            self.coral_head = CoralHead(dim, self.num_coral, dropout=dropout)
        else:
            self.encoder_k = copy.deepcopy(self.encoder_q)
            for p in self.encoder_k.parameters():
                p.requires_grad_(False)
            self.queue = FeatureQueue(dim=dim, size=queue_size)
            self.head = MoCoFusionHead(dim=dim, num_coral=self.num_coral, dropout=dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_details: bool = False,
        enqueue_keys: bool = True,
    ):
        f_q = self.encoder_q(x_q)
        if self.head_mode == "cls":
            logits = self.cls_head(f_q)
            if return_details:
                return logits, f_q, None, None, None, None
            return logits
        if self.head_mode == "coral":
            logits = self.coral_head(f_q)
            if return_details:
                return logits, f_q, None, None, None, None
            return logits

        f_moco, alpha = self.queue.read(f_q, self.read_temperature)
        logits = self.head(f_q, f_moco)

        f_k = None
        if self.training and x_k is not None:
            with torch.no_grad():
                f_k = self.encoder_k(x_k)
            if labels is not None and enqueue_keys:
                self.queue.enqueue(f_k, labels)

        if return_details:
            return logits, f_q, f_k, f_moco, alpha, self.queue.get_valid()[0]
        return logits


def apply_backbone_freeze_moco(model: DRMocoModel, freeze: bool) -> int:
    if model.head_mode != "coral_moco":
        enc = model.encoder_q
        if not freeze:
            for p in enc.parameters():
                p.requires_grad = True
        else:
            for name, p in enc.named_parameters():
                p.requires_grad = not name.startswith("backbone.")
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    if not freeze:
        for p in model.encoder_q.parameters():
            p.requires_grad = True
    else:
        for name, p in model.encoder_q.named_parameters():
            p.requires_grad = not name.startswith("backbone.")
    for p in model.encoder_k.parameters():
        p.requires_grad = False
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class DRTwoViewDataset(Dataset):
    """同一张图生成 query（强增强）与 key（弱增强）视图。"""

    def __init__(self, images_root: str, list_file: str, transform_q, transform_k):
        self.base = APTOSDrDataset(images_root, list_file, transform=None)
        self.transform_q = transform_q
        self.transform_k = transform_k

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        path, label = self.base.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform_q(img), self.transform_k(img), label


# ---------------------------------------------------------------------------
# MoCo loss
# ---------------------------------------------------------------------------


def supervised_moco_loss(
    f_q: torch.Tensor,
    f_k: torch.Tensor,
    labels: torch.Tensor,
    queue_feats: torch.Tensor,
    queue_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    监督 MoCo InfoNCE：正样本 = 动量 key + 队列中同类；负样本 = 队列中异类。
    """
    labels = labels.long()
    pos_logit = (f_q * f_k).sum(dim=1, keepdim=True) / float(temperature)

    if queue_feats.numel() == 0:
        return F.softplus(-pos_logit.squeeze(1)).mean()

    neg_logits = (f_q @ queue_feats.T) / float(temperature)
    logits = torch.cat([pos_logit, neg_logits], dim=1)

    pos_mask = torch.zeros_like(logits, dtype=torch.bool)
    pos_mask[:, 0] = True
    if neg_logits.size(1) > 0:
        pos_mask[:, 1:] = queue_labels.unsqueeze(0) == labels.unsqueeze(1)

    log_denom = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = logits - log_denom
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
    n_pos = pos_mask.float().sum(dim=1).clamp_min(1.0)
    return (-pos_log_prob / n_pos).mean()


def mixup_minority_dual_views(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    minority_classes: set[int],
    mixup_prob: float,
    mixup_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """对 query/key 两视图用相同 λ/perm 做少数类 Mixup。"""
    device = x_q.device
    b = x_q.size(0)
    base_coral = labels_to_coral_targets(labels, num_classes)
    base_probs = labels_to_class_probs(labels, num_classes)
    mix_mask = torch.zeros(b, dtype=torch.bool, device=device)
    if mixup_prob <= 0 or not minority_classes:
        return x_q, x_k, base_coral, base_probs, mix_mask

    is_minority = torch.zeros(num_classes, dtype=torch.bool, device=device)
    for c in minority_classes:
        if 0 <= c < num_classes:
            is_minority[c] = True
    minority_mask = is_minority[labels]
    perm = torch.randperm(b, device=device)
    lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample((b,)).to(device)
    rand_apply = torch.rand(b, device=device) < mixup_prob
    mix_mask = minority_mask & rand_apply

    lam_4d = lam.view(b, 1, 1, 1)
    mixed_q = lam_4d * x_q + (1.0 - lam_4d) * x_q[perm]
    mixed_k = lam_4d * x_k + (1.0 - lam_4d) * x_k[perm]
    mix_4d = mix_mask.view(b, 1, 1, 1)
    x_q_out = torch.where(mix_4d, mixed_q, x_q)
    x_k_out = torch.where(mix_4d, mixed_k, x_k)

    lam_y = lam.view(b, 1)
    target_coral = torch.where(
        mix_mask.unsqueeze(1).expand(-1, num_classes - 1),
        lam_y * base_coral + (1.0 - lam_y) * base_coral[perm],
        base_coral,
    )
    target_probs = torch.where(
        mix_mask.unsqueeze(1).expand(-1, num_classes),
        lam_y * base_probs + (1.0 - lam_y) * base_probs[perm],
        base_probs,
    )
    return x_q_out, x_k_out, target_coral, target_probs, mix_mask


def get_moco_weight(max_weight: float, epoch: int, warmup_epochs: int) -> float:
    if max_weight <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return float(max_weight)
    return float(max_weight) * min(float(epoch) / float(warmup_epochs), 1.0)


@torch.no_grad()
def refresh_queue(
    model: DRMocoModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_batches: Optional[int] = None,
) -> int:
    """用训练集弱增强视图 + encoder_k 填满队列（验证/测试前调用）。"""
    if model.head_mode != "coral_moco":
        return 0
    model.eval()
    model.queue.reset()
    n_enqueued = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if len(batch) == 3:
            _xq, xk, labels = batch
        else:
            xk, labels = batch[0], batch[1]
        xk = xk.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            f_k = model.encoder_k(xk)
        model.queue.enqueue(f_k, labels)
        n_enqueued += int(labels.size(0))
    model.train()
    print(f"[moco] queue refreshed: {model.queue.num_valid()} entries (enqueued {n_enqueued})")
    return n_enqueued


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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
    parser.add_argument("--dropout_rate", type=float, default=0.3)
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
    parser.add_argument("--out_dir", type=str, default="./outputs_ordinal_moco")
    parser.add_argument("--log_csv", type=str, default="training_log.csv")
    parser.add_argument("--no_strong_aug", action="store_true")
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--mixup_prob", type=float, default=0.5)
    parser.add_argument("--mixup_num_rare", type=int, default=3)
    parser.add_argument("--constant_lr", action="store_true")
    parser.add_argument("--lanet_dynamic_lr", action="store_true")
    parser.add_argument("--lanet_lr_power", type=float, default=0.9)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--cosine_eta_min", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_eval", action="store_true")
    parser.add_argument("--gaussian_blur_sigma", type=float, default=0.5)
    parser.add_argument("--gaussian_blur_kernel", type=int, default=5)
    parser.add_argument("--emd_weight", type=float, default=0.3)
    parser.add_argument(
        "--moco_weight",
        type=float,
        default=0.1,
        help="监督 MoCo 对比损失权重；0 关闭",
    )
    parser.add_argument("--moco_warmup_epochs", type=int, default=5)
    parser.add_argument("--moco_momentum", type=float, default=0.999)
    parser.add_argument("--moco_temperature", type=float, default=0.07)
    parser.add_argument("--moco_queue_size", type=int, default=4096)
    parser.add_argument(
        "--read_temperature",
        type=float,
        default=0.07,
        help="队列检索 softmax 温度",
    )
    parser.add_argument(
        "--refresh_queue_eval",
        dest="refresh_queue_eval",
        action="store_true",
        help="验证/测试前用训练集弱增强刷新队列（默认开启）",
    )
    parser.add_argument(
        "--no_refresh_queue_eval",
        dest="refresh_queue_eval",
        action="store_false",
        help="验证/测试不刷新队列，使用训练中累积的队列",
    )
    parser.set_defaults(refresh_queue_eval=True)
    parser.add_argument(
        "--refresh_queue_batches",
        type=int,
        default=0,
        help="刷新队列最多用多少个 batch（0=全训练集）",
    )
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="argmax",
        choices=["argmax", "expectation"],
    )
    parser.add_argument(
        "--head_mode",
        type=str,
        default="coral_moco",
        choices=["cls", "coral", "coral_moco"],
    )
    args = parser.parse_args()

    if args.eval_only and not args.checkpoint:
        raise SystemExit("--eval_only 需要同时指定 --checkpoint")
    if args.moco_temperature <= 0 or args.read_temperature <= 0:
        raise SystemExit("moco_temperature 与 read_temperature 必须 > 0")
    if args.head_mode == "cls" and args.emd_weight > 0:
        raise SystemExit("head_mode=cls 时请设 --emd_weight 0")
    if args.head_mode != "coral_moco" and args.moco_weight > 0:
        raise SystemExit("moco_weight>0 仅适用于 head_mode=coral_moco")
    if args.constant_lr and args.lanet_dynamic_lr:
        raise SystemExit("--constant_lr 与 --lanet_dynamic_lr 不能同时使用")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits_dir = Path(args.splits_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = 5

    model = DRMocoModel(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=not args.eval_only,
        dropout=args.dropout_rate,
        queue_size=args.moco_queue_size,
        read_temperature=args.read_temperature,
        head_mode=args.head_mode,
    ).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)

    n_trainable = apply_backbone_freeze_moco(model, args.freeze_backbone)
    print(f"[model] head_mode={args.head_mode} | queue_size={args.moco_queue_size}")

    train_tf_q = build_transforms(
        args.img_size,
        train=True,
        strong_aug=not args.no_strong_aug,
        gaussian_blur_sigma=args.gaussian_blur_sigma,
        gaussian_blur_kernel=args.gaussian_blur_kernel,
    )
    train_tf_k = build_transforms(
        args.img_size,
        train=True,
        strong_aug=False,
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

    def make_eval_loader(list_name, shuffle):
        ds = APTOSDrDataset(args.images_root, str(splits_dir / list_name), transform=eval_tf)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    def make_train_two_view_loader():
        ds = DRTwoViewDataset(
            args.images_root,
            str(splits_dir / args.train_list),
            transform_q=train_tf_q,
            transform_k=train_tf_k,
        )
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    def make_key_refresh_loader():
        ds = APTOSDrDataset(
            args.images_root, str(splits_dir / args.train_list), transform=train_tf_k
        )
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    use_amp = device.type == "cuda"
    refresh_batches = args.refresh_queue_batches if args.refresh_queue_batches > 0 else None

    def eval_with_queue(eval_model, loader):
        if args.head_mode == "coral_moco" and args.refresh_queue_eval:
            refresh_queue(
                eval_model,
                make_key_refresh_loader(),
                device,
                use_amp,
                max_batches=refresh_batches,
            )
        return evaluate_ordinal_metrics_and_loss(
            eval_model,
            loader,
            device,
            num_classes,
            use_amp,
            head_mode=args.head_mode,
            emd_weight=args.emd_weight,
            decode_mode=args.decode_mode,
        )

    if args.eval_only:
        split_map = {
            "train": ("train", args.train_list),
            "val": ("val", args.val_list),
            "test": ("test", args.test_list),
        }
        order = ["train", "val", "test"] if args.eval_sets == "all" else [args.eval_sets]
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key in order:
            name, fname = split_map[key]
            loader = make_eval_loader(fname, shuffle=False)
            m, _ = eval_with_queue(model, loader)
            print(f"[eval_only] {name}: QWK={m['qwk']:.4f} Acc={m['acc']:.4f} F1={m['f1_macro']:.4f}")
            yt, yp = collect_predictions(
                model, loader, device, num_classes, use_amp,
                head_mode=args.head_mode, decode_mode=args.decode_mode,
            )
            if args.head_mode == "coral_moco" and args.refresh_queue_eval:
                refresh_queue(model, make_key_refresh_loader(), device, use_amp, refresh_batches)
            print_and_save_confusion_matrix(yt, yp, num_classes, out_dir, split_name=key)
        return

    train_loader = (
        make_train_two_view_loader()
        if args.head_mode == "coral_moco"
        else make_eval_loader(args.train_list, shuffle=True)
    )
    valid_loader = make_eval_loader(args.val_list, shuffle=False)
    test_loader = make_eval_loader(args.test_list, shuffle=False)

    train_list_path = splits_dir / args.train_list
    class_counts = count_class_frequencies(train_list_path, num_classes)
    rare_indices = pick_rarest_class_indices(class_counts, args.mixup_num_rare)
    minority_classes = set(rare_indices) if not args.no_mixup else set()
    use_mixup = bool(minority_classes) and args.mixup_prob > 0 and not args.no_mixup
    use_moco = args.moco_weight > 0.0 and args.head_mode == "coral_moco"
    print(
        f"[train] 类别计数 {class_counts} | Mixup={sorted(minority_classes) if use_mixup else '关'} | "
        f"EMD={args.emd_weight} | MoCo w={args.moco_weight} warmup={args.moco_warmup_epochs} | "
        f"queue={args.moco_queue_size} m={args.moco_momentum} | "
        f"refresh_queue_eval={args.refresh_queue_eval}"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lanet_dynamic_lr:
        apply_optimizer_lr(optimizer, lanet_dynamic_lr(0, args.epochs, args.lr, args.lanet_lr_power))
    elif not args.constant_lr:
        eta_min = (
            args.cosine_eta_min if args.cosine_eta_min is not None else max(1e-6, float(args.lr) * 0.01)
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer, args.epochs, args.warmup_epochs, eta_min
        )

    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / args.log_csv
    log_fields = [
        "epoch", "train_loss", "train_moco_loss", "moco_weight",
        "val_loss", "val_qwk", "val_acc", "val_f1_macro",
        "val_sensitivity_macro", "val_specificity_macro", "queue_size",
    ]
    best_qwk = -1.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        if args.lanet_dynamic_lr:
            apply_optimizer_lr(
                optimizer, lanet_dynamic_lr(epoch - 1, args.epochs, args.lr, args.lanet_lr_power)
            )
        model.train()
        total_loss = 0.0
        total_moco_loss = 0.0
        moco_weight = get_moco_weight(args.moco_weight, epoch, args.moco_warmup_epochs)

        for batch in train_loader:
            if args.head_mode == "coral_moco":
                x_q, x_k, labels = batch
                x_q = x_q.to(device, non_blocking=True)
                x_k = x_k.to(device, non_blocking=True)
            else:
                x_q, labels = batch
                x_q = x_q.to(device, non_blocking=True)
                x_k = None
            labels = labels.to(device, non_blocking=True)
            mix_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=device)

            if args.head_mode == "cls":
                if use_mixup:
                    x_q, target_cls_soft = mixup_minority_batch_cls(
                        x_q, labels, num_classes, minority_classes,
                        args.mixup_prob, args.mixup_alpha,
                    )
                    use_cls_soft = True
                else:
                    target_cls_soft = None
                    use_cls_soft = False
                target_coral = labels_to_coral_targets(labels, num_classes)
                target_probs = labels_to_class_probs(labels, num_classes)
            elif use_mixup and args.head_mode == "coral_moco" and x_k is not None:
                x_q, x_k, target_coral, target_probs, mix_mask = mixup_minority_dual_views(
                    x_q, x_k, labels, num_classes, minority_classes,
                    args.mixup_prob, args.mixup_alpha,
                )
            elif use_mixup:
                x_q, target_coral, target_probs, mix_mask = mixup_minority_batch_coral(
                    x_q, labels, num_classes, minority_classes,
                    args.mixup_prob, args.mixup_alpha,
                )
            else:
                target_coral = labels_to_coral_targets(labels, num_classes)
                target_probs = labels_to_class_probs(labels, num_classes)
                use_cls_soft = False
                target_cls_soft = None

            queue_feats, queue_labels = (
                model.queue.get_valid() if args.head_mode == "coral_moco" else (None, None)
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if args.head_mode == "coral_moco":
                    logits, f_q, f_k, _fm, _alpha, _qf = model(
                        x_q,
                        x_k=x_k,
                        labels=labels,
                        return_details=True,
                        enqueue_keys=False,
                    )
                else:
                    logits = model(x_q)
                    f_q = f_k = None

                if args.head_mode == "cls" and use_mixup:
                    loss = compute_training_loss(
                        args.head_mode, logits, labels, target_coral, target_probs,
                        target_cls_soft, args.emd_weight, True,
                    )
                else:
                    loss = compute_training_loss(
                        args.head_mode, logits, labels, target_coral, target_probs,
                        None, args.emd_weight, False,
                    )

                moco_loss = torch.zeros((), device=device)
                if (
                    use_moco
                    and moco_weight > 0
                    and f_k is not None
                    and queue_feats is not None
                ):
                    clean_mask = ~mix_mask
                    if int(clean_mask.sum()) > 0:
                        moco_loss = supervised_moco_loss(
                            f_q[clean_mask],
                            f_k[clean_mask],
                            labels[clean_mask],
                            queue_feats,
                            queue_labels,
                            args.moco_temperature,
                        )
                        loss = loss + moco_weight * moco_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.head_mode == "coral_moco" and f_k is not None:
                model.queue.enqueue(f_k.detach(), labels)
                momentum_update_encoder(
                    model.encoder_q, model.encoder_k, args.moco_momentum
                )
            if ema is not None:
                ema.update(model)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_moco_loss += moco_loss.item() * bs

        if scheduler is not None:
            scheduler.step()

        n = len(train_loader.dataset)
        train_loss = total_loss / max(n, 1)
        train_moco_loss = total_moco_loss / max(n, 1)
        queue_n = model.queue.num_valid() if args.head_mode == "coral_moco" else 0

        eval_model = ema.ema if (args.ema_eval and ema is not None) else model
        valid_m, val_loss = eval_with_queue(eval_model, valid_loader)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{args.epochs} lr={lr_now:.2e} "
            f"loss={train_loss:.4f} moco={train_moco_loss:.4f} w={moco_weight:.3f} "
            f"queue={queue_n} val_QWK={valid_m['qwk']:.4f} Acc={valid_m['acc']:.4f}"
        )

        write_header = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=log_fields)
            if write_header:
                w.writeheader()
            w.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_moco_loss": train_moco_loss,
                "moco_weight": moco_weight,
                "val_loss": val_loss,
                "val_qwk": valid_m["qwk"],
                "val_acc": valid_m["acc"],
                "val_f1_macro": valid_m["f1_macro"],
                "val_sensitivity_macro": valid_m["sensitivity_macro"],
                "val_specificity_macro": valid_m["specificity_macro"],
                "queue_size": queue_n,
            })

        if valid_m["qwk"] > best_qwk + args.early_stop_min_delta:
            best_qwk = valid_m["qwk"]
            torch.save({
                "model": model.state_dict(),
                "ema_model": ema.ema.state_dict() if ema is not None else None,
                "epoch": epoch,
                "val_metrics": valid_m,
                "head": args.head_mode,
                "args": vars(args),
            }, out_dir / "best.pt")
            print(f"  saved best.pt (QWK={best_qwk:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                print(f"[early_stop] best QWK={best_qwk:.4f}")
                break

    best_path = out_dir / "best.pt"
    if best_path.is_file():
        loaded = torch.load(best_path, map_location=device)
        model.load_state_dict(loaded["model"])
        if args.ema_eval and ema is not None and loaded.get("ema_model"):
            ema.ema.load_state_dict(loaded["ema_model"], strict=True)

    final_model = ema.ema if (args.ema_eval and ema is not None) else model
    test_m, test_loss = eval_with_queue(final_model, test_loader)
    print(f"\n[test] QWK={test_m['qwk']:.4f} Acc={test_m['acc']:.4f} F1={test_m['f1_macro']:.4f}")

    yt_val, yp_val = collect_predictions(
        final_model, valid_loader, device, num_classes, use_amp,
        head_mode=args.head_mode, decode_mode=args.decode_mode,
    )
    if args.head_mode == "coral_moco" and args.refresh_queue_eval:
        refresh_queue(final_model, make_key_refresh_loader(), device, use_amp, refresh_batches)
    print_and_save_confusion_matrix(yt_val, yp_val, num_classes, out_dir, "val")

    yt_test, yp_test = collect_predictions(
        final_model, test_loader, device, num_classes, use_amp,
        head_mode=args.head_mode, decode_mode=args.decode_mode,
    )
    if args.head_mode == "coral_moco" and args.refresh_queue_eval:
        refresh_queue(final_model, make_key_refresh_loader(), device, use_amp, refresh_batches)
    print_and_save_confusion_matrix(yt_test, yp_test, num_classes, out_dir, "test")

    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "val": compute_dr_metrics(yt_val, yp_val, num_classes=num_classes),
            "test": compute_dr_metrics(yt_test, yp_test, num_classes=num_classes),
            "test_loss": float(test_loss),
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
