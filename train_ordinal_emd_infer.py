#!/usr/bin/env python3
"""
与 train_ordinal.py 相同（训练、损失、数据、选 best 流程一致），
仅在推理/验证/测试的类别解码上使用与 EMD 训练一致的 5 类概率：

  logits → coral_probs_monotonic → p_pred (B, K)
  → argmax 或 round(E[p]) 得到 0..K-1

原 train_ordinal.py 推理为 CORAL 阈值计数（sigmoid>0.5 个数），与本脚本不同。

用法示例：
  # 仅评测（需已训练 checkpoint）
  python train_ordinal_emd_infer.py --eval_only --checkpoint ./outputs_ordinal_emd0.3_dropout0.3/best.pt \\
    --emd_weight 0.3 --ema_eval --decode_mode argmax

  # 完整训练（验证/测试指标按 EMD 解码统计）
  python train_ordinal_emd_infer.py --emd_weight 0.3 --ema_eval --decode_mode argmax \\
    --out_dir ./outputs_ordinal_emd0.3_emd_infer
"""
from __future__ import annotations

import argparse
import sys

import torch

import train_ordinal as base


def emd_predict(
    logits: torch.Tensor,
    num_classes: int,
    decode_mode: str = "argmax",
) -> torch.Tensor:
    """
    EMD 一致推理：与训练时 compute_ordinal_loss 中的 p_pred 同源。

    decode_mode:
      - argmax: 取单调 5 类概率最大类（默认）
      - expectation: round(Σ k * p_k)，有序期望等级
    """
    p_pred = base.coral_probs_monotonic(logits)
    if decode_mode == "argmax":
        return p_pred.argmax(dim=1)
    if decode_mode == "expectation":
        k = torch.arange(num_classes, device=p_pred.device, dtype=p_pred.dtype)
        return (p_pred * k.unsqueeze(0)).sum(dim=1).round().long().clamp(0, num_classes - 1)
    raise ValueError(f"Unknown decode_mode: {decode_mode!r}")


def patch_coral_predict(decode_mode: str) -> None:
    """将 base.coral_predict 替换为 EMD 解码（仅影响 evaluate / collect_predictions）。"""

    def _predict(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
        return emd_predict(logits, num_classes, decode_mode=decode_mode)

    base.coral_predict = _predict  # type: ignore[method-assign]


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--decode_mode",
        type=str,
        default="argmax",
        choices=["argmax", "expectation"],
        help="EMD 单调 5 类概率的解码方式：argmax 或 round(期望等级)",
    )
    pre_args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    patch_coral_predict(pre_args.decode_mode)
    print(
        f"[inference] EMD-consistent decode: coral_probs_monotonic → {pre_args.decode_mode} "
        f"(训练损失仍用 train_ordinal.compute_ordinal_loss)"
    )
    base.main()


if __name__ == "__main__":
    main()
