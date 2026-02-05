"""
Bottleneck VAE Trainer.

Trains a BottleneckVAE on pre-encoded SD VAE latents (4x16x16).
The bottleneck compresses to a flat 256-d vector where MM-Reg is applied.

Pipeline:
    1. Pre-encode images with frozen SD VAE -> (N, 4, 16, 16)
    2. Train BottleneckVAE: 4x16x16 -> 256-d -> 4x16x16
    3. MM-Reg enforces structure on the 256-d space
    4. Diffusion then trains on the 256-d vectors
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Any
from tqdm import tqdm
import json
from pathlib import Path

from .models.bottleneck import BottleneckVAE, BottleneckLoss


class LatentPCADataset(TensorDataset):
    """Dataset of (SD latent, PCA embedding) pairs."""

    def __init__(self, latents: torch.Tensor, pca_embeddings: torch.Tensor):
        """
        Args:
            latents: SD VAE latents (N, 4, 16, 16)
            pca_embeddings: PCA reference embeddings (N, D_pca)
        """
        super().__init__(latents, pca_embeddings)

    def __getitem__(self, idx):
        return self.tensors[0][idx], self.tensors[1][idx]


class LatentPCAAttrsDataset(TensorDataset):
    """Dataset of (SD latent, PCA embedding, attributes) triples."""

    def __init__(
        self,
        latents: torch.Tensor,
        pca_embeddings: torch.Tensor,
        attributes: torch.Tensor
    ):
        super().__init__(latents, pca_embeddings, attributes)

    def __getitem__(self, idx):
        return self.tensors[0][idx], self.tensors[1][idx], self.tensors[2][idx]


def encode_dataset_to_sd_latents(
    vae: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Encode entire dataset to SD VAE latents.

    Args:
        vae: Trained MMRegVAE model (SD VAE wrapper)
        dataloader: DataLoader for images
        device: Device to use

    Returns:
        Tensor of all latents (N, 4, 16, 16)
    """
    vae.eval()
    all_latents = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding to SD latents"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            outputs = vae(images, sample=False)
            latents = outputs['latent']  # (B, 4, H, W)
            all_latents.append(latents.cpu())

    return torch.cat(all_latents, dim=0)


class BottleneckTrainer:
    """
    Trainer for the BottleneckVAE.

    Trains on pre-encoded SD VAE latents with:
    - MSE reconstruction loss on the 4x16x16 latent space
    - KL divergence on the 256-d bottleneck posterior
    - MM-Reg on the 256-d bottleneck vectors (optional, controlled by lambda_mm)
    """

    def __init__(
        self,
        bottleneck: BottleneckVAE,
        loss_fn: BottleneckLoss,
        optimizer: torch.optim.Optimizer,
        train_latents: torch.Tensor,
        train_pca: torch.Tensor,
        val_latents: Optional[torch.Tensor] = None,
        val_pca: Optional[torch.Tensor] = None,
        batch_size: int = 128,
        device: str = 'cuda',
        use_amp: bool = True,
        log_interval: int = 50,
        save_dir: str = './checkpoints/bottleneck',
        scheduler: Optional[Any] = None
    ):
        """
        Args:
            bottleneck: BottleneckVAE model
            loss_fn: BottleneckLoss instance
            optimizer: Optimizer
            train_latents: SD VAE latents for training (N, 4, 16, 16)
            train_pca: PCA reference embeddings for training (N, D_pca)
            val_latents: SD VAE latents for validation
            val_pca: PCA reference embeddings for validation
            batch_size: Batch size
            device: Training device
            use_amp: Use automatic mixed precision
            log_interval: Steps between logging
            save_dir: Directory for checkpoints
            scheduler: Learning rate scheduler
        """
        self.bottleneck = bottleneck
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.scheduler = scheduler
        self.batch_size = batch_size

        # Create dataloaders
        self.train_loader = DataLoader(
            LatentPCADataset(train_latents, train_pca),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        self.val_loader = None
        if val_latents is not None and val_pca is not None:
            self.val_loader = DataLoader(
                LatentPCADataset(val_latents, val_pca),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )

        # Mixed precision
        self.scaler = GradScaler() if use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # History
        self.train_history = []
        self.val_history = []

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.bottleneck.train()
        epoch_metrics = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'mm_loss': 0.0
        }
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, (latents, pca_emb) in enumerate(pbar):
            latents = latents.to(self.device)
            pca_emb = pca_emb.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.bottleneck(latents)
                    losses = self.loss_fn(
                        x=latents,
                        x_recon=outputs['x_recon'],
                        z=outputs['z'],
                        r=pca_emb,
                        mu=outputs['mu'],
                        logvar=outputs['logvar']
                    )

                self.scaler.scale(losses['loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.bottleneck(latents)
                losses = self.loss_fn(
                    x=latents,
                    x_recon=outputs['x_recon'],
                    z=outputs['z'],
                    r=pca_emb,
                    mu=outputs['mu'],
                    logvar=outputs['logvar']
                )

                losses['loss'].backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            for key in epoch_metrics:
                epoch_metrics[key] += losses[key].item()
            num_batches += 1
            self.global_step += 1

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{losses['loss'].item():.4f}",
                    'recon': f"{losses['recon_loss'].item():.4f}",
                    'mm': f"{losses['mm_loss'].item():.4f}"
                })

        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.bottleneck.eval()
        val_metrics = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'mm_loss': 0.0
        }
        num_batches = 0

        for latents, pca_emb in self.val_loader:
            latents = latents.to(self.device)
            pca_emb = pca_emb.to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.bottleneck(latents, sample=False)
                    losses = self.loss_fn(
                        x=latents,
                        x_recon=outputs['x_recon'],
                        z=outputs['z'],
                        r=pca_emb,
                        mu=outputs['mu'],
                        logvar=outputs['logvar']
                    )
            else:
                outputs = self.bottleneck(latents, sample=False)
                losses = self.loss_fn(
                    x=latents,
                    x_recon=outputs['x_recon'],
                    z=outputs['z'],
                    r=pca_emb,
                    mu=outputs['mu'],
                    logvar=outputs['logvar']
                )

            for key in val_metrics:
                val_metrics[key] += losses[key].item()
            num_batches += 1

        for key in val_metrics:
            val_metrics[key] /= num_batches

        return val_metrics

    def train(self, num_epochs: int, save_every: int = 5):
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs
            save_every: Save checkpoint every N epochs
        """
        print(f"Training BottleneckVAE for {num_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Bottleneck dim: {self.bottleneck.latent_dim}")
        print(f"Device: {self.device}, AMP: {self.use_amp}")
        print(f"Loss weights: beta={self.loss_fn.beta}, lambda_mm={self.loss_fn.lambda_mm}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            print(f"\nEpoch {epoch} - Train: "
                  f"loss={train_metrics['loss']:.4f}, "
                  f"recon={train_metrics['recon_loss']:.4f}, "
                  f"kl={train_metrics['kl_loss']:.2f}, "
                  f"mm={train_metrics['mm_loss']:.4f}")

            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)

                print(f"Epoch {epoch} - Val: "
                      f"loss={val_metrics['loss']:.4f}, "
                      f"recon={val_metrics['recon_loss']:.4f}, "
                      f"kl={val_metrics['kl_loss']:.2f}, "
                      f"mm={val_metrics['mm_loss']:.4f}")

                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best.pt')

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')

        self.save_checkpoint('final.pt')
        self.save_history()
        print("Training complete!")

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'bottleneck_state_dict': self.bottleneck.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': {
                'latent_dim': self.bottleneck.latent_dim,
                'spatial_shape': self.bottleneck.spatial_shape
            }
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.save_dir / filename)
        print(f"Saved checkpoint: {self.save_dir / filename}")

    def load_checkpoint(self, filename: str):
        """Load checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)

        self.bottleneck.load_state_dict(checkpoint['bottleneck_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from epoch {self.epoch}")

    def save_history(self):
        """Save training history."""
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    @torch.no_grad()
    def encode_all(self, latents: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """
        Encode all SD latents to bottleneck vectors.

        Args:
            latents: SD VAE latents (N, 4, 16, 16)
            batch_size: Batch size for encoding

        Returns:
            Bottleneck vectors (N, latent_dim)
        """
        self.bottleneck.eval()
        all_z = []

        for i in range(0, len(latents), batch_size):
            batch = latents[i:i + batch_size].to(self.device)
            enc = self.bottleneck.encode(batch, sample=False)
            all_z.append(enc['mu'].cpu())

        return torch.cat(all_z, dim=0)
