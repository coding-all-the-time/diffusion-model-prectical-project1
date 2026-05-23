"""
Dataset 加载器（完整参考实现，无需修改）

支持 MNIST、CIFAR-10。学生不需要修改本文件，但建议读懂归一化逻辑。

⚠️ 关键：DDPM 期望输入 x ∈ [-1, 1]，需要把图像从 [0, 1] 映射到 [-1, 1]。
"""

import os
from typing import Tuple, Callable
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF


def get_transform(image_size: int, dataset_name: str = 'cifar10') -> Callable:
    """
    返回标准的训练 transform。

    步骤：
        1. 转 tensor（自动归一化到 [0, 1]）
        2. 缩放/裁剪到 image_size（仅当需要）
        3. 随机水平翻转（CIFAR-10 常用，MNIST 不用）
        4. 归一化到 [-1, 1]
    """
    transform_list = [transforms.ToTensor()]  # → [0, 1]

    # MNIST 是 28x28，可能需要 padding 到 32x32 以便下采样
    if dataset_name.lower() == 'mnist' and image_size != 28:
        transform_list.append(transforms.Resize(image_size))
    elif dataset_name.lower() == 'cifar10' and image_size != 32:
        transform_list.append(transforms.Resize(image_size))

    # 数据增强（仅训练）
    if dataset_name.lower() in ['cifar10']:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # 归一化到 [-1, 1]
    if dataset_name.lower() == 'mnist':
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        transform_list.append(transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    return transforms.Compose(transform_list)


class DiffusionDataset(Dataset):
    """
    Wrapper：返回的是 image tensor，扔掉 label（DDPM 是无条件生成）。
    """

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]  # 丢弃 label
        return x


def get_dataset(
    name: str,
    root: str = './data',
    image_size: int = None,
    train: bool = True,
) -> Dataset:
    """
    构造数据集。

    Args:
        name: 'mnist' or 'cifar10'
        root: 数据存储根目录
        image_size: 图像分辨率（默认按数据集决定）
        train: 是否使用训练集

    Returns:
        Dataset 实例（每个元素是 (C, H, W) 的 tensor，值在 [-1, 1]）
    """
    name = name.lower()
    if image_size is None:
        image_size = 32 if name == 'cifar10' else 28

    transform = get_transform(image_size, name)

    if name == 'mnist':
        base = datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )
    elif name == 'cifar10':
        base = datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return DiffusionDataset(base)


def get_dataloader(
    name: str,
    batch_size: int,
    root: str = './data',
    image_size: int = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    构造 DataLoader（标准训练用配置）。
    """
    dataset = get_dataset(name, root=root, image_size=image_size, train=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


# ============================================================================
# 反归一化（用于可视化）
# ============================================================================
def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]，用于显示或保存图像。"""
    return (x.clamp(-1, 1) + 1) / 2


# ============================================================================
# 自测
# ============================================================================
if __name__ == '__main__':
    print("=== Testing dataset.py ===")

    # 测试 MNIST
    try:
        loader = get_dataloader('mnist', batch_size=4, num_workers=0)
        x = next(iter(loader))
        assert x.shape == (4, 1, 28, 28), f"Expected (4,1,28,28), got {x.shape}"
        print(f"  MNIST: shape={x.shape}, range=[{x.min():.2f}, {x.max():.2f}]")
        assert -1.01 <= x.min() and x.max() <= 1.01, "Should be in [-1, 1]"
        print("✅ MNIST loader passed")
    except Exception as e:
        print(f"⚠️ MNIST loader: {e}")

    # 测试 CIFAR-10
    try:
        loader = get_dataloader('cifar10', batch_size=4, num_workers=0)
        x = next(iter(loader))
        assert x.shape == (4, 3, 32, 32)
        print(f"  CIFAR-10: shape={x.shape}, range=[{x.min():.2f}, {x.max():.2f}]")
        print("✅ CIFAR-10 loader passed")
    except Exception as e:
        print(f"⚠️ CIFAR-10 loader: {e}")
