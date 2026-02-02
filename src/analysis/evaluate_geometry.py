"""
Geometric evaluation metrics for MM-Reg.

Measures:
1. Distance Correlation: Pearson correlation between latent and reference distances
2. Linear Probing: Classification accuracy with linear probe on frozen latents
3. Reconstruction Quality: PSNR, SSIM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances."""
    dot_product = torch.mm(x, x.t())
    square_norm = dot_product.diag()
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0)
    return torch.sqrt(distances + 1e-8)


def compute_distance_correlation(
    vae: nn.Module,
    reference_model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    max_samples: int = 5000
) -> Dict[str, float]:
    """
    Compute correlation between latent and reference pairwise distances.

    Args:
        vae: VAE model
        reference_model: Reference extractor
        dataloader: Data loader
        device: Device
        max_samples: Maximum number of samples to use

    Returns:
        Dictionary with pearson and spearman correlations
    """
    vae.eval()
    all_latents = []
    all_refs = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting features"):
            if len(all_latents) * images.shape[0] >= max_samples:
                break

            images = images.to(device)

            # Get latents
            outputs = vae(images, sample=False)
            latents = outputs['latent_flat']

            # Get references
            refs = reference_model(images)

            all_latents.append(latents.cpu())
            all_refs.append(refs.cpu())

    # Concatenate
    all_latents = torch.cat(all_latents, dim=0)[:max_samples]
    all_refs = torch.cat(all_refs, dim=0)[:max_samples]

    # Compute pairwise distances
    D_latent = pairwise_distances(all_latents)
    D_ref = pairwise_distances(all_refs)

    # Extract upper triangular
    n = D_latent.shape[0]
    indices = torch.triu_indices(n, n, offset=1)
    d_latent = D_latent[indices[0], indices[1]].numpy()
    d_ref = D_ref[indices[0], indices[1]].numpy()

    # Compute correlations
    pearson_corr, _ = pearsonr(d_latent, d_ref)
    spearman_corr, _ = spearmanr(d_latent, d_ref)

    return {
        'pearson_correlation': float(pearson_corr),
        'spearman_correlation': float(spearman_corr)
    }


class LinearProbe(nn.Module):
    """Simple linear classifier for probing."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_probe(
    vae: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: str = 'cuda',
    epochs: int = 10,
    lr: float = 0.01
) -> Dict[str, float]:
    """
    Train linear probe on frozen VAE latents.

    Args:
        vae: VAE model (frozen)
        train_loader: Training data
        val_loader: Validation data
        num_classes: Number of classes
        device: Device
        epochs: Training epochs
        lr: Learning rate

    Returns:
        Dictionary with train and val accuracy
    """
    vae.eval()

    # First, extract all latents
    print("Extracting train latents...")
    train_latents, train_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            outputs = vae(images, sample=False)
            train_latents.append(outputs['latent_flat'].cpu())
            train_labels.append(labels)

    train_latents = torch.cat(train_latents, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    print("Extracting val latents...")
    val_latents, val_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            outputs = vae(images, sample=False)
            val_latents.append(outputs['latent_flat'].cpu())
            val_labels.append(labels)

    val_latents = torch.cat(val_latents, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    # Create and train probe
    latent_dim = train_latents.shape[1]
    probe = LinearProbe(latent_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Move data to device
    train_latents = train_latents.to(device)
    train_labels = train_labels.to(device)
    val_latents = val_latents.to(device)
    val_labels = val_labels.to(device)

    # Train
    batch_size = 256
    best_val_acc = 0.0

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(train_latents.shape[0])

        for i in range(0, train_latents.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            batch_x = train_latents[idx]
            batch_y = train_labels[idx]

            optimizer.zero_grad()
            logits = probe(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            train_logits = probe(train_latents)
            train_acc = (train_logits.argmax(1) == train_labels).float().mean().item()

            val_logits = probe(val_latents)
            val_acc = (val_logits.argmax(1) == val_labels).float().mean().item()

        best_val_acc = max(best_val_acc, val_acc)

    return {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'best_val_accuracy': best_val_acc
    }


def compute_reconstruction_metrics(
    vae: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    max_batches: int = 50
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.

    Args:
        vae: VAE model
        dataloader: Data loader
        device: Device
        max_batches: Maximum batches to evaluate

    Returns:
        Dictionary with PSNR and SSIM
    """
    vae.eval()
    all_psnr = []
    all_mse = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Computing reconstruction")):
            if batch_idx >= max_batches:
                break

            images = images.to(device)

            # Reconstruct
            outputs = vae(images, sample=False)
            recon = outputs['x_recon']

            # Convert from [-1, 1] to [0, 1]
            images_01 = (images + 1) / 2
            recon_01 = (recon + 1) / 2
            recon_01 = torch.clamp(recon_01, 0, 1)

            # MSE
            mse = F.mse_loss(recon_01, images_01, reduction='none').mean(dim=[1, 2, 3])
            all_mse.extend(mse.cpu().tolist())

            # PSNR
            psnr = -10 * torch.log10(mse + 1e-8)
            all_psnr.extend(psnr.cpu().tolist())

    return {
        'psnr': float(np.mean(all_psnr)),
        'psnr_std': float(np.std(all_psnr)),
        'mse': float(np.mean(all_mse))
    }


def full_evaluation(
    vae: nn.Module,
    reference_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Run full geometric evaluation.

    Args:
        vae: VAE model
        reference_model: Reference extractor
        train_loader: Training data
        val_loader: Validation data
        num_classes: Number of classes
        device: Device

    Returns:
        Dictionary with all metrics
    """
    results = {}

    print("\n" + "=" * 50)
    print("1. Distance Correlation")
    print("=" * 50)
    corr_metrics = compute_distance_correlation(
        vae, reference_model, val_loader, device
    )
    results.update(corr_metrics)
    print(f"Pearson: {corr_metrics['pearson_correlation']:.4f}")
    print(f"Spearman: {corr_metrics['spearman_correlation']:.4f}")

    print("\n" + "=" * 50)
    print("2. Linear Probing")
    print("=" * 50)
    probe_metrics = train_linear_probe(
        vae, train_loader, val_loader, num_classes, device
    )
    results.update({f'probe_{k}': v for k, v in probe_metrics.items()})
    print(f"Val Accuracy: {probe_metrics['val_accuracy']:.4f}")

    print("\n" + "=" * 50)
    print("3. Reconstruction Quality")
    print("=" * 50)
    recon_metrics = compute_reconstruction_metrics(vae, val_loader, device)
    results.update({f'recon_{k}': v for k, v in recon_metrics.items()})
    print(f"PSNR: {recon_metrics['psnr']:.2f} dB")
    print(f"MSE: {recon_metrics['mse']:.6f}")

    return results
