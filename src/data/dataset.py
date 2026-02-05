"""
Data loading utilities for MM-Reg.

Supports:
- Imagenette (small subset for quick testing)
- ImageNet (full training)
- CIFAR-10 (for debugging)
- CelebA (faces with 40 attributes for interpolation experiments)

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


# CelebA attribute names for reference
CELEBA_ATTRIBUTES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]


class CelebADataset(Dataset):
    """
    CelebA dataset wrapper that returns (image, attributes).

    Attributes are a tensor of 40 binary values (0 or 1).
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        download: bool = True
    ):
        # Map split names
        split_map = {'train': 'train', 'val': 'valid', 'test': 'test'}
        celeba_split = split_map.get(split, split)

        self.dataset = datasets.CelebA(
            root=root,
            split=celeba_split,
            target_type='attr',
            transform=transform,
            download=download
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, attrs = self.dataset[idx]
        # Convert attributes from -1/1 to 0/1 if needed, and to float
        attrs = attrs.float()
        attrs = (attrs + 1) / 2  # CelebA uses -1/1, convert to 0/1
        return image, attrs


class CelebAHuggingFace(Dataset):
    """
    CelebA dataset loaded from HuggingFace.

    Avoids Google Drive rate limits.
    The HF dataset only has a 'train' split, so we manually split:
    - train: first 162,770 images (standard CelebA train split)
    - val: next 19,867 images (standard CelebA val split)
    - test: last 19,962 images (standard CelebA test split)
    """
    def __init__(
        self,
        split: str = 'train',
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        from datasets import load_dataset

        print(f"Loading CelebA from HuggingFace (split={split})...")
        # HF only has 'train' split with all 202,599 images
        full_dataset = load_dataset("nielsr/CelebA-faces", split="train")

        # Standard CelebA partition sizes
        train_end = 162770
        val_end = train_end + 19867  # 182637

        if split == 'train':
            self.dataset = full_dataset.select(range(train_end))
        elif split == 'val':
            self.dataset = full_dataset.select(range(train_end, val_end))
        elif split == 'test':
            self.dataset = full_dataset.select(range(val_end, len(full_dataset)))
        else:
            self.dataset = full_dataset

        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        self.transform = transform

        # HuggingFace CelebA-faces doesn't include attributes
        # We'll return dummy attributes (zeros) - for full attributes use CelebAFromDrive
        self.has_attributes = False
        print(f"Loaded {len(self.dataset)} images for split '{split}'")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']

        if self.transform:
            image = self.transform(image)

        # Return dummy attributes (40 zeros) since HF version doesn't have them
        attrs = torch.zeros(40)
        return image, attrs


class CelebAFromDrive(Dataset):
    """
    CelebA dataset loaded from Google Drive (pre-downloaded).

    Expects:
    - images_zip: Path to img_align_celeba.zip
    - attr_file: Path to list_attr_celeba.txt
    - partition_file: Path to list_eval_partition.txt
    """
    def __init__(
        self,
        images_zip: str,
        attr_file: str,
        partition_file: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        extract_dir: str = './data/celeba_extracted'
    ):
        import zipfile
        from PIL import Image

        self.transform = transform
        self.extract_dir = extract_dir

        # Extract images if needed
        img_dir = os.path.join(extract_dir, 'img_align_celeba')
        if not os.path.exists(img_dir):
            print(f"Extracting {images_zip}...")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("Extraction complete!")

        self.img_dir = img_dir

        # Load attributes
        print("Loading attributes...")
        self.attrs_dict = {}
        with open(attr_file, 'r') as f:
            lines = f.readlines()
            # First line is count, second line is header
            header = lines[1].strip().split()
            for line in lines[2:]:
                parts = line.strip().split()
                filename = parts[0]
                attrs = [1 if int(x) == 1 else 0 for x in parts[1:]]
                self.attrs_dict[filename] = torch.tensor(attrs, dtype=torch.float32)

        # Load partition
        print("Loading partition...")
        split_map = {'train': 0, 'val': 1, 'test': 2}
        target_split = split_map.get(split, 0)

        self.image_files = []
        with open(partition_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, partition = parts[0], int(parts[1])
                    if partition == target_split:
                        self.image_files.append(filename)

        print(f"Loaded {len(self.image_files)} images for split '{split}'")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        from PIL import Image

        filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, filename)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        attrs = self.attrs_dict.get(filename, torch.zeros(40))
        return image, attrs


class PCAEmbeddingDatasetWithAttributes(Dataset):
    """
    Dataset that returns images with attributes AND pre-computed PCA embeddings.
    Returns (image, attributes, pca_embedding) tuples.
    """
    def __init__(
        self,
        base_dataset: Dataset,
        pca_embeddings_path: str
    ):
        self.dataset = base_dataset
        self.pca_embeddings = torch.load(pca_embeddings_path)

        assert len(self.pca_embeddings) == len(self.dataset), \
            f"PCA embeddings ({len(self.pca_embeddings)}) must match dataset size ({len(self.dataset)})"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, attrs = self.dataset[idx]
        pca_emb = self.pca_embeddings[idx]
        return image, attrs, pca_emb


def get_celeba_dataset(
    root: str = './data',
    split: str = 'train',
    image_size: int = 128,
    fixed_transform: bool = False,
    download: bool = True,
    source: str = 'auto',
    drive_paths: Optional[dict] = None,
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Load CelebA dataset from various sources.

    Args:
        root: Data directory
        split: 'train', 'val', or 'test'
        image_size: Target image size (default 128 for faces)
        fixed_transform: Use fixed transforms (no augmentation)
        download: Download if not present
        source: 'auto', 'torchvision', 'huggingface', or 'drive'
        drive_paths: Dict with 'images_zip', 'attr_file', 'partition_file' for Drive source
        max_samples: Limit number of samples (useful for testing)

    Returns:
        Dataset that returns (image, attributes)
    """
    if fixed_transform:
        transform = get_fixed_transforms(image_size)
    else:
        transform = get_train_transforms(image_size) if split == 'train' else get_val_transforms(image_size)

    # Try sources in order
    if source == 'auto':
        sources_to_try = ['torchvision', 'huggingface']
        if drive_paths:
            sources_to_try = ['drive'] + sources_to_try
    else:
        sources_to_try = [source]

    last_error = None
    for src in sources_to_try:
        try:
            if src == 'drive' and drive_paths:
                print(f"Loading CelebA from Google Drive...")
                dataset = CelebAFromDrive(
                    images_zip=drive_paths['images_zip'],
                    attr_file=drive_paths['attr_file'],
                    partition_file=drive_paths['partition_file'],
                    split=split,
                    transform=transform,
                    extract_dir=os.path.join(root, 'celeba_extracted')
                )
                return dataset

            elif src == 'huggingface':
                print(f"Loading CelebA from HuggingFace...")
                dataset = CelebAHuggingFace(
                    split=split,
                    transform=transform,
                    max_samples=max_samples
                )
                return dataset

            elif src == 'torchvision':
                print(f"Loading CelebA from torchvision...")
                dataset = CelebADataset(
                    root=root,
                    split=split,
                    transform=transform,
                    download=download
                )
                return dataset

        except Exception as e:
            print(f"Failed to load from {src}: {e}")
            last_error = e
            continue

    raise RuntimeError(f"Could not load CelebA from any source. Last error: {last_error}")


def compute_pca_embeddings_celeba(
    dataset: Dataset,
    n_components: int = 256,
    batch_size: int = 64,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Compute PCA embeddings for CelebA dataset.

    Similar to compute_pca_embeddings but handles (image, attrs) format.
    """
    from sklearn.decomposition import IncrementalPCA

    pca_batch_size = max(batch_size, n_components)
    print(f"Using batch_size={pca_batch_size} for PCA (n_components={n_components})")

    loader = DataLoader(dataset, batch_size=pca_batch_size, shuffle=False, num_workers=4)

    print("Fitting PCA...")
    pca = IncrementalPCA(n_components=n_components)

    for images, _ in loader:  # Ignore attributes for PCA
        flat = images.view(images.shape[0], -1).numpy()
        pca.partial_fit(flat)

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
    pca_embeddings_path: Optional[str] = None,
    celeba_source: str = 'huggingface',
    celeba_drive_paths: Optional[dict] = None
) -> Tuple[Dataset, DataLoader]:
    """
    Get dataset and dataloader, optionally with pre-computed PCA embeddings.

    Args:
        dataset_name: 'imagenette', 'cifar10', 'imagenet', or 'celeba'
        root: Data directory
        split: 'train' or 'val'/'test'
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of workers
        pca_embeddings_path: Path to pre-computed PCA embeddings (optional)
        celeba_source: Source for CelebA ('huggingface', 'drive', 'torchvision')
        celeba_drive_paths: Drive paths for CelebA if using drive source

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
    elif dataset_name == 'celeba':
        dataset = get_celeba_dataset(
            root, split, image_size,
            fixed_transform=fixed_transform,
            source=celeba_source,
            drive_paths=celeba_drive_paths
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Wrap with PCA embeddings if provided
    if pca_embeddings_path is not None:
        if dataset_name == 'celeba':
            dataset = PCAEmbeddingDatasetWithAttributes(dataset, pca_embeddings_path)
        else:
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
