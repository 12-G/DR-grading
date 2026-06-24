from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def make_loader(images_root, splits_path, is_train=True, batch_size=32, num_workers=8, shuffle=None, transform=None,
                img_size=384, gaussian_blur_sigma=0.5, gaussian_blur_kernel=5):
    if transform is None:
        if is_train:
            transform = build_transforms(
                img_size,
                train=True,
                strong_aug=False,
                gaussian_blur_sigma=gaussian_blur_sigma,
                gaussian_blur_kernel=gaussian_blur_kernel,
            )
        else:
            transform = build_transforms(
                img_size,
                train=False,
                strong_aug=False,
                gaussian_blur_sigma=gaussian_blur_sigma,
                gaussian_blur_kernel=gaussian_blur_kernel,
            )
    shuffle = shuffle if shuffle is not None else is_train

    ds = BaseDataset(images_root, splits_path, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
