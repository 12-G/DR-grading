#!/usr/bin/env python3
"""
按原有 splits list 做 fundus 圆盘离线裁剪，输出目录与 list 中相对路径一致。

这样 train/val/test 仍用 AOR-DR 的 APTOS_train_80.txt 等，无需改 list；
训练时只需把 --images_root 从原始 APTOS2019 换成裁剪后的根目录。

示例:
  cd fundus_circle_cropping && PYTHONPATH=. python ../crop_from_splits.py \\
    --src_root /root/autodl-tmp/baseline/AOR-DR/data/APTOS2019 \\
    --dst_root /root/autodl-tmp/APTOS2019_circle1024 \\
    --splits_dir /root/autodl-tmp/baseline/AOR-DR/data/splits
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_FCC_ROOT = Path(__file__).resolve().parent / "fundus_circle_cropping"
if str(_FCC_ROOT) not in sys.path:
    sys.path.insert(0, str(_FCC_ROOT))

from fundus_circle_cropping import fundus_cropping  # noqa: E402

DEFAULT_LISTS = (
    "APTOS_train_80.txt",
    "APTOS_val_20.txt",
    "APTOS_crossval.txt",
)


def normalize_rel(path_in_list: str) -> str:
    rel = path_in_list.replace("\\", "/")
    if rel.startswith("APTOS/"):
        rel = rel[len("APTOS/") :]
    return rel


def collect_paths_from_splits(splits_dir: Path, list_names: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in list_names:
        list_path = splits_dir / name
        if not list_path.is_file():
            raise FileNotFoundError(f"Missing split file: {list_path}")
        with open(list_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel = normalize_rel(line.split()[0])
                if rel not in seen:
                    seen.add(rel)
                    ordered.append(rel)
    return ordered


def main():
    parser = argparse.ArgumentParser(
        description="Fundus circle crop using existing split lists (paths unchanged)"
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default="/root/autodl-tmp/baseline/AOR-DR/data/APTOS2019",
        help="原始图像根目录（须与 splits 中 nodr/pdr/... 路径一致）",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="/root/autodl-tmp/APTOS2019_circle1024",
        help="裁剪后输出根目录（内部仍保留 nodr/pdr/... 子目录）",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/root/autodl-tmp/baseline/AOR-DR/data/splits",
    )
    parser.add_argument(
        "--lists",
        type=str,
        nargs="+",
        default=list(DEFAULT_LISTS),
        help="要处理的 split 文件名（会去重合并）",
    )
    parser.add_argument("--resize_shape", type=int, default=1024)
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--fit_largest_contour", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="试跑：只处理前 N 张")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    splits_dir = Path(args.splits_dir).resolve()

    rel_paths = collect_paths_from_splits(splits_dir, tuple(args.lists))
    print(f"[info] unique images from splits: {len(rel_paths)}")

    failures: list[str] = []
    missing_src: list[str] = []
    ok = skipped = 0

    for i, rel in enumerate(rel_paths):
        if args.limit > 0 and i >= args.limit:
            break

        src_path = src_root / rel
        if not src_path.is_file():
            missing_src.append(rel)
            continue

        out_dir = dst_root / Path(rel).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(rel).stem
        out_png = out_dir / f"{stem}.png"

        if args.skip_existing and out_png.is_file():
            skipped += 1
            continue

        mask_dir = dst_root / "masks" / Path(rel).parent
        if args.save_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)
        else:
            mask_dir.mkdir(parents=True, exist_ok=True)

        try:
            x = np.array(Image.open(src_path).convert("RGB"))
            _, fail = fundus_cropping.fundus_image(
                x=x,
                x_id=stem,
                image_folder=str(out_dir),
                file_extension="png",
                mask_folder=str(mask_dir),
                resize_shape=args.resize_shape,
                remove_rectangles=True,
                fit_largest_contour=args.fit_largest_contour,
            )
            if fail:
                failures.append(rel)
            else:
                ok += 1
                if ok % 200 == 0:
                    print(f"[progress] ok={ok} fail={len(failures)} skipped={skipped}")
        except Exception as e:
            failures.append(rel)
            print(f"[ERROR] {rel}: {e}")

    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "failures.lst").write_text("\n".join(failures), encoding="utf-8")
    (dst_root / "missing_src.lst").write_text("\n".join(missing_src), encoding="utf-8")

    print(
        f"\nDone. ok={ok} skipped={skipped} crop_fail={len(failures)} "
        f"missing_src={len(missing_src)} -> {dst_root}"
    )
    if missing_src:
        print(f"  源图缺失列表: {dst_root / 'missing_src.lst'}")
    if failures:
        print(f"  裁剪失败列表: {dst_root / 'failures.lst'}")


if __name__ == "__main__":
    main()
