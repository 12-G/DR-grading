from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    每行: `<相对路径> <标签>`，相对路径可能以 `APTOS/` 或 `DDR/` 开头；
    实际根目录为 `images_root`，会自动去掉对应前缀。
    """

    def __init__(self, images_root: str, list_file: str, transform=None):
        self.images_root = Path(images_root).resolve()
        self.transform = transform
        self.samples = []
        missing = 0
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Bad line in {list_file}: {line}")
                rel = parts[0].replace("\\", "/")
                label = int(parts[1])
                is_ddr = rel.startswith("DDR/")
                if rel.startswith("APTOS/"):
                    rel = rel[len("APTOS/") :]
                if is_ddr:
                    rel = rel[len("DDR/") :]
                full = self.images_root / rel
                # DDR 兼容：划分文件为 DDR/moderate_npdr/xxx.jpg，本机常为 0..4/<filename>
                if (not full.is_file()) and is_ddr:
                    fname = Path(rel).name
                    full = self.images_root / str(label) / fname
                    if not full.is_file():
                        for cls_dir in ("0", "1", "2", "3", "4"):
                            cand = self.images_root / cls_dir / fname
                            if cand.is_file():
                                full = cand
                                break
                # EyePACS 兼容：部分划分文件写成 EYEPACS/Images/xxx.jpeg，
                # 但本机数据为 images_root/<label>/xxx.jpeg（按类别分目录）。
                if (not full.is_file()) and rel.startswith("EYEPACS/Images/"):
                    full = self.images_root / str(label) / Path(rel).name
                # 二次回退：若标签目录仍不存在该文件，则在 0..4 目录里按文件名搜索
                # （用于 split 标签与本机目录划分不一致的情况）。
                if (not full.is_file()) and rel.startswith("EYEPACS/Images/"):
                    fname = Path(rel).name
                    for cls_dir in ("0", "1", "2", "3", "4"):
                        cand = self.images_root / cls_dir / fname
                        if cand.is_file():
                            full = cand
                            break
                # crossval 兼容：部分 EyePACS 测试图位于 images_root/test/<filename>
                if (not full.is_file()) and rel.startswith("EYEPACS/Images/"):
                    fname = Path(rel).name
                    # 场景1：images_root=/root/.../EyePACS（测试图在其 test 子目录）
                    cand_a = self.images_root / "test" / fname
                    # 场景2：images_root 已直接指向 /root/.../EyePACS/test
                    cand_b = self.images_root / fname
                    if cand_a.is_file():
                        full = cand_a
                    elif cand_b.is_file():
                        full = cand_b
                if not full.is_file():
                    missing += 1
                    if missing <= 20:
                        print(f"[dataset] warning: missing image skipped: {full}")
                    continue
                self.samples.append((str(full), label))
        if missing > 0:
            print(f"[dataset] total missing skipped from {list_file}: {missing}")
        if not self.samples:
            raise RuntimeError(f"No valid samples loaded from {list_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
