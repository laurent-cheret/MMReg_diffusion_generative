"""
Data loading utilities for MM-Reg.

Supports:
- Imagenette (small subset for quick testing)
- ImageNet (full training)
- CIFAR-10 (for debugging)
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from typing import Optional, Tuple, Callable
import os


def get_train_transforms(image_size: int = 256) -> Callable:
    """
    Get training transforms with data augmentation.

    Args:
        image_size: Target image size

    Returns:
        Transform composition
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> [-1, 1]
    ])


def get_val_transforms(image_size: int = 256) -> Callable:
    """
    Get validation transforms (no augmentation).

    Args:
        image_size: Target image size

    Returns:
        Transform composition
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> [-1, 1]
    ])


def get_imagenette_dataset(
    root: str = './data',
    split: str = 'train',
    image_size: int = 256,
    download: bool = True
) -> Dataset:
    """
    Load Imagenette dataset (10-class subset of ImageNet).

    Good for quick validation and debugging.

    Args:
        root: Data directory
        split: 'train' or 'val'
        image_size: Target image size
        download: Download if not present

    Returns:
        Dataset
    """
    transform = get_train_transforms(image_size) if split == 'train' else get_val_transforms(image_size)

    # Imagenette is available via HuggingFace datasets or torchvision
    # Using a simple approach with ImageFolder
    dataset_path = os.path.join(root, 'imagenette2', split)

    if not os.path.exists(dataset_path):
        if download:
            print("Downloading Imagenette...")
            import urllib.request
            import tarfile

            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
            tar_path = os.path.join(root, "imagenette2.tgz")

            os.makedirs(root, exist_ok=True)
            urllib.request.urlretrieve(url, tar_path)

            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(root)

            os.remove(tar_path)
        else:
            raise RuntimeError(f"Dataset not found at {dataset_path}")

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    return dataset


def get_cifar10_dataset(
    root: str = './data',
    split: str = 'train',
    image_size: int = 256
) -> Dataset:
    """
    Load CIFAR-10 dataset (for debugging).

    Args:
        root: Data directory
        split: 'train' or 'test'
        image_size: Target image size

    Returns:
        Dataset
    """
    transform = get_train_transforms(image_size) if split == 'train' else get_val_transforms(image_size)

    dataset = datasets.CIFAR10(
        root=root,
        train=(split == 'train'),
        transform=transform,
        download=True
    )
    return dataset


def get_imagenet_dataset(
    root: str,
    split: str = 'train',
    image_size: int = 256
) -> Dataset:
    """
    Load ImageNet dataset.

    Requires manual download of ImageNet.

    Args:
        root: Path to ImageNet directory (containing train/ and val/)
        split: 'train' or 'val'
        image_size: Target image size

    Returns:
        Dataset
    """
    transform = get_train_transforms(image_size) if split == 'train' else get_val_transforms(image_size)

    dataset_path = os.path.join(root, split)
    if not os.path.exists(dataset_path):
        raise RuntimeError(f"ImageNet not found at {dataset_path}")

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    return dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    Create DataLoader from dataset.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )


def get_dataset_and_loader(
    dataset_name: str = 'imagenette',
    root: str = './data',
    split: str = 'train',
    image_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 4
) -> Tuple[Dataset, DataLoader]:
    """
    Convenience function to get dataset and dataloader.

    Args:
        dataset_name: 'imagenette', 'cifar10', or 'imagenet'
        root: Data directory
        split: 'train' or 'val'/'test'
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        Tuple of (dataset, dataloader)
    """
    if dataset_name == 'imagenette':
        dataset = get_imagenette_dataset(root, split, image_size)
    elif dataset_name == 'cifar10':
        dataset = get_cifar10_dataset(root, split, image_size)
    elif dataset_name == 'imagenet':
        dataset = get_imagenet_dataset(root, split, image_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    shuffle = (split == 'train')
    loader = get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle
    )

    return dataset, loader
