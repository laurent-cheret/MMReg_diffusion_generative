"""
Reference Embedding Extractors for MM-Reg.

Provides frozen feature extractors:
- DINOv2: Semantic features from vision transformer
- PCA: Linear projection capturing dominant variations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class DINOv2Reference(nn.Module):
    """
    DINOv2 feature extractor for reference embeddings.

    Uses the CLS token from DINOv2-ViT-B/14 as semantic reference.
    All parameters are frozen.
    """

    def __init__(self, model_name: str = 'dinov2_vitb14', device: str = 'cuda'):
        """
        Args:
            model_name: DINOv2 model variant
            device: Device to load model on
        """
        super().__init__()
        self.device = device

        # Load DINOv2 from torch hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # DINOv2 normalization (ImageNet stats)
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv2 CLS token features.

        Args:
            x: Images in range [0, 1] or [-1, 1], shape (B, 3, H, W)

        Returns:
            Feature vectors (B, 768) for ViT-B
        """
        # Ensure input is in [0, 1]
        if x.min() < 0:
            x = (x + 1) / 2

        # Resize to DINOv2 expected size (224x224 or 518x518 for large)
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        # Extract CLS token
        features = self.model(x)

        return features


class PCAReference(nn.Module):
    """
    PCA-based reference extractor.

    Projects images to top-k principal components computed from dataset.
    """

    def __init__(
        self,
        n_components: int = 256,
        image_size: int = 256,
        pca_path: Optional[str] = None
    ):
        """
        Args:
            n_components: Number of PCA components
            image_size: Expected input image size
            pca_path: Path to pre-computed PCA matrix (optional)
        """
        super().__init__()
        self.n_components = n_components
        self.image_size = image_size

        # PCA projection matrix (D_flat, n_components)
        # Will be set by fit() or load()
        self.register_buffer('components', None)
        self.register_buffer('mean', None)

        if pca_path is not None:
            self.load(pca_path)

    def fit(self, images: torch.Tensor, max_samples: int = 50000):
        """
        Fit PCA on a batch of images.

        Args:
            images: Tensor of shape (N, C, H, W) in range [0, 1]
            max_samples: Maximum number of samples to use
        """
        # Subsample if needed
        if images.shape[0] > max_samples:
            indices = torch.randperm(images.shape[0])[:max_samples]
            images = images[indices]

        # Flatten images
        flat = images.view(images.shape[0], -1).float()  # (N, C*H*W)

        # Center the data
        mean = flat.mean(dim=0)
        flat_centered = flat - mean

        # Compute PCA via SVD
        # For large matrices, use randomized SVD or incremental PCA
        U, S, Vt = torch.linalg.svd(flat_centered, full_matrices=False)

        # Take top k components
        components = Vt[:self.n_components].T  # (D_flat, n_components)

        self.register_buffer('components', components)
        self.register_buffer('mean', mean)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project images to PCA space.

        Args:
            x: Images in range [0, 1] or [-1, 1], shape (B, C, H, W)

        Returns:
            PCA features (B, n_components)
        """
        if self.components is None:
            raise RuntimeError("PCA not fitted. Call fit() or load() first.")

        # Ensure input is in [0, 1]
        if x.min() < 0:
            x = (x + 1) / 2

        # Resize if needed
        if x.shape[-1] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )

        # Flatten
        flat = x.view(x.shape[0], -1)

        # Project
        flat_centered = flat - self.mean.to(x.device)
        features = torch.mm(flat_centered, self.components.to(x.device))

        return features

    def save(self, path: str):
        """Save PCA parameters to file."""
        torch.save({
            'components': self.components,
            'mean': self.mean,
            'n_components': self.n_components,
            'image_size': self.image_size
        }, path)

    def load(self, path: str):
        """Load PCA parameters from file."""
        data = torch.load(path, map_location='cpu')
        self.register_buffer('components', data['components'])
        self.register_buffer('mean', data['mean'])
        self.n_components = data['n_components']
        self.image_size = data['image_size']


class HybridReference(nn.Module):
    """
    Combines DINOv2 and PCA references.

    Returns concatenated features from both extractors.
    """

    def __init__(
        self,
        dino_model: str = 'dinov2_vitb14',
        pca_components: int = 256,
        pca_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.dino = DINOv2Reference(model_name=dino_model, device=device)
        self.pca = PCAReference(n_components=pca_components, pca_path=pca_path)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Extract both DINOv2 and PCA features.

        Returns:
            Tuple of (dino_features, pca_features)
        """
        dino_feat = self.dino(x)
        pca_feat = self.pca(x) if self.pca.components is not None else None
        return dino_feat, pca_feat


def get_reference_model(
    ref_type: str = 'dinov2',
    device: str = 'cuda',
    pca_path: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to get reference model.

    Args:
        ref_type: 'dinov2', 'pca', or 'hybrid'
        device: Device to load model on
        pca_path: Path to pre-computed PCA (for pca/hybrid)

    Returns:
        Reference model
    """
    if ref_type == 'dinov2':
        return DINOv2Reference(device=device, **kwargs)
    elif ref_type == 'pca':
        return PCAReference(pca_path=pca_path, **kwargs)
    elif ref_type == 'hybrid':
        return HybridReference(pca_path=pca_path, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown reference type: {ref_type}")
