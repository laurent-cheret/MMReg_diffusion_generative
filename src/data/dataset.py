"""
Data loading utilities for MM-Reg.

Supports:
- Imagenette (small subset for quick testing)
- ImageNet (full training)
- CIFAR-10 (for debugging)

For MM-Reg training, we need:
1. Fixed transforms (no random augmentation) for PCA pre-computation
2. Dataset that returns sample indices for looking up pre-computed PCA embeddings
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from typing import Optional, Tuple, Callable
import os
import numpy as np


def get_fixed_transforms(image_size: int = 256) -> Callable:
    """
    Get fixed transforms (no augmentation) for PCA pre-computation.
    Must be deterministic so we can cache PCA embeddings.
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_train_transforms(image_size: int = 256) -> Callable:
    """Get training transforms with data augmentation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_val_transforms(image_size: int = 256) -> Callable:
    """Get validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


class IndexedDataset(Dataset):
    """
    Wrapper that adds sample indices to any dataset.
    Returns (image, label, index) instead of (image, label).
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, idx


class PCAEmbeddingDataset(Dataset):
    """
    Dataset that returns images with their pre-computed PCA embeddings.

    Usage:
    1. Pre-compute PCA embeddings using compute_pca_embeddings()
    2. Load this dataset with the path to saved embeddings
    3. During training, get (image, label, pca_embedding) tuples
    """
    def __init__(
        self,
        base_dataset: Dataset,
        pca_embeddings_path: str
    ):
        """
        Args:
            base_dataset: The underlying image dataset
            pca_embeddings_path: Path to .pt file with pre-computed PCA embeddings
        """
        self.dataset = base_dataset
        self.pca_embeddings = torch.load(pca_embeddings_path)

        assert len(self.pca_embeddings) == len(self.dataset), \
            f"PCA embeddings ({len(self.pca_embeddings)}) must match dataset size ({len(self.dataset)})"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        pca_emb = self.pca_embeddings[idx]
        return image, label, pca_emb


def compute_pca_embeddings(
    dataset: Dataset,
    n_components: int = 256,
    batch_size: int = 64,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Compute PCA embeddings for all samples in a dataset.

    Args:
        dataset: Dataset to compute embeddings for (should use fixed transforms!)
        n_components: Number of PCA components (can be any value up to min(n_samples, n_features))
        batch_size: Batch size for processing (will be adjusted if < n_components)
        device: Device to use

    Returns:
        Tensor of shape (N, n_components) with PCA embeddings
    """
    from sklearn.decomposition import IncrementalPCA

    # IncrementalPCA requires batch_size >= n_components
    # Adjust batch_size if needed
    pca_batch_size = max(batch_size, n_components)
    print(f"Using batch_size={pca_batch_size} for PCA (n_components={n_components})")

    loader = DataLoader(dataset, batch_size=pca_batch_size, shuffle=False, num_workers=4)

    # First pass: fit PCA incrementally
    print("Fitting PCA...")
    pca = IncrementalPCA(n_components=n_components)

    for images, _ in loader:
        # Flatten images: (B, C, H, W) -> (B, C*H*W)
        flat = images.view(images.shape[0], -1).numpy()
        pca.partial_fit(flat)

    # Second pass: transform all samples (can use smaller batches)
    print("Transforming samples...")
    transform_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_embeddings = []

    for images, _ in transform_loader:
        flat = images.view(images.shape[0], -1).numpy()
        emb = pca.transform(flat)
        all_embeddings.append(torch.from_numpy(emb).float())

    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"PCA embeddings shape: {embeddings.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    return embeddings


def get_imagenette_dataset(
    root: str = './data',
    split: str = 'train',
    image_size: int = 256,
    download: bool = True,
    fixed_transform: bool = False
) -> Dataset:
    """Load Imagenette dataset."""
    if fixed_transform:
        transform = get_fixed_transforms(image_size)
    else:
        transform = get_train_transforms(image_size) if split == 'train' else get_val_transforms(image_size)

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
    image_size: int = 256,
    fixed_transform: bool = False
) -> Dataset:
    """Load CIFAR-10 dataset."""
    if fixed_transform:
        transform = get_fixed_transforms(image_size)
    else:
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
    image_size: int = 256,
    fixed_transform: bool = False
) -> Dataset:
    """Load ImageNet dataset."""
    if fixed_transform:
        transform = get_fixed_transforms(image_size)
    else:
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
    """Create DataLoader from dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0 if num_workers > 0 else False
    )


def get_dataset_and_loader(
    dataset_name: str = 'imagenette',
    root: str = './data',
    split: str = 'train',
    image_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 4,
    pca_embeddings_path: Optional[str] = None
) -> Tuple[Dataset, DataLoader]:
    """
    Get dataset and dataloader, optionally with pre-computed PCA embeddings.

    Args:
        dataset_name: 'imagenette', 'cifar10', or 'imagenet'
        root: Data directory
        split: 'train' or 'val'/'test'
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of workers
        pca_embeddings_path: Path to pre-computed PCA embeddings (optional)

    Returns:
        Tuple of (dataset, dataloader)
    """
    # Use fixed transforms if we have PCA embeddings (they were computed with fixed transforms)
    fixed_transform = pca_embeddings_path is not None

    if dataset_name == 'imagenette':
        dataset = get_imagenette_dataset(root, split, image_size, fixed_transform=fixed_transform)
    elif dataset_name == 'cifar10':
        dataset = get_cifar10_dataset(root, split, image_size, fixed_transform=fixed_transform)
    elif dataset_name == 'imagenet':
        dataset = get_imagenet_dataset(root, split, image_size, fixed_transform=fixed_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Wrap with PCA embeddings if provided
    if pca_embeddings_path is not None:
        dataset = PCAEmbeddingDataset(dataset, pca_embeddings_path)

    shuffle = (split == 'train')
    loader = get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle
    )

    return dataset, loader
