#!/usr/bin/env python3
"""从 APTOS2019_circle1024 的 0..4 目录生成 train/val/test list（分层随机划分）。"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_samples(root: Path) -> dict[int, list[str]]:
    by_label: dict[int, list[str]] = {i: [] for i in range(5)}
    for label in range(5):
        d = root / str(label)
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_SUFFIXES:
                # list 格式：<相对路径> <label>
                rel = f"{label}/{p.name}"
                by_label[label].append(f"{rel} {label}")
    return by_label


def stratified_split(
    by_label: dict[int, list[str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for label in range(5):
        items = by_label[label][:]
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_root",
        type=str,
        default="/root/autodl-tmp/APTOS2019_circle1024",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/root/autodl-tmp/APTOS2019_circle1024/splits",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.images_root)
    by_label = collect_samples(root)
    total = sum(len(v) for v in by_label.values())
    if total == 0:
        raise SystemExit(f"No images under {root}/0..4")

    tr = args.train_ratio
    vr = args.val_ratio
    if tr + vr >= 1.0:
        raise SystemExit("train_ratio + val_ratio 必须 < 1（留一部分给 test）")

    train, val, test = stratified_split(by_label, tr, vr, args.seed)
    out = Path(args.splits_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "APTOS_train_80.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "APTOS_val_20.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    (out / "APTOS_crossval.txt").write_text("\n".join(test) + "\n", encoding="utf-8")

    print(f"total={total} train={len(train)} val={len(val)} test={len(test)}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
