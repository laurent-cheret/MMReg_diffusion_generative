"""
Diffusion Model Trainer.

Trains a diffusion model on pre-encoded VAE latents.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Any, List
from tqdm import tqdm
import json
from pathlib import Path


class LatentDataset(TensorDataset):
    """Dataset of pre-encoded latents."""

    def __init__(self, latents: torch.Tensor):
        super().__init__(latents)

    def __getitem__(self, idx):
        return self.tensors[0][idx]


def encode_dataset(
    vae: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Encode entire dataset to latents using VAE.

    Args:
        vae: Trained VAE model
        dataloader: DataLoader for images
        device: Device to use

    Returns:
        Tensor of all latents (N, 4, 32, 32)
    """
    vae.eval()
    all_latents = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding dataset"):
            # Handle both (images,) and (images, labels) and (images, labels, pca)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            outputs = vae(images, sample=False)
            latents = outputs['latent']  # (B, 4, 32, 32)
            all_latents.append(latents.cpu())

    return torch.cat(all_latents, dim=0)


class DiffusionTrainer:
    """
    Trainer for latent diffusion model.
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion,  # GaussianDiffusion
        optimizer: torch.optim.Optimizer,
        train_latents: torch.Tensor,
        val_latents: Optional[torch.Tensor] = None,
        batch_size: int = 64,
        device: str = 'cuda',
        use_amp: bool = True,
        log_interval: int = 100,
        save_dir: str = './checkpoints/diffusion',
        scheduler: Optional[Any] = None
    ):
        """
        Args:
            model: Diffusion UNet model
            diffusion: GaussianDiffusion instance
            optimizer: Optimizer
            train_latents: Pre-encoded training latents (N, 4, 32, 32)
            val_latents: Pre-encoded validation latents
            batch_size: Batch size
            device: Training device
            use_amp: Use automatic mixed precision
            log_interval: Steps between logging
            save_dir: Directory for checkpoints
            scheduler: Learning rate scheduler
        """
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.scheduler = scheduler
        self.batch_size = batch_size

        # Store latent shape for generation (infer from training data)
        self.latent_shape = train_latents.shape[1:]  # (C, H, W)

        # Create dataloaders
        self.train_loader = DataLoader(
            LatentDataset(train_latents),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        self.val_loader = None
        if val_latents is not None:
            self.val_loader = DataLoader(
                LatentDataset(val_latents),
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
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, latents in enumerate(pbar):
            latents = latents.to(self.device)
            batch_size = latents.shape[0]

            # Sample random timesteps
            t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    loss = self.diffusion.p_losses(self.model, latents, t)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.diffusion.p_losses(self.model, latents, t)
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return {'loss': total_loss / num_batches}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for latents in self.val_loader:
            latents = latents.to(self.device)
            batch_size = latents.shape[0]

            t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)

            if self.use_amp:
                with autocast():
                    loss = self.diffusion.p_losses(self.model, latents, t)
            else:
                loss = self.diffusion.p_losses(self.model, latents, t)

            total_loss += loss.item()
            num_batches += 1

        return {'loss': total_loss / num_batches}

    def train(self, num_epochs: int, save_every: int = 5, sample_every: int = 10):
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs
            save_every: Save checkpoint every N epochs
            sample_every: Generate samples every N epochs
        """
        print(f"Training diffusion model for {num_epochs} epochs")
        print(f"Training latents: {len(self.train_loader.dataset)}")
        print(f"Device: {self.device}, AMP: {self.use_amp}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            log_msg = f"Epoch {epoch} - Train loss: {train_metrics['loss']:.4f}"

            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)
                log_msg += f", Val loss: {val_metrics['loss']:.4f}"

                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best.pt')

            print(log_msg)

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
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.save_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']

    def save_history(self):
        """Save training history."""
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    @torch.no_grad()
    def generate_samples(self, num_samples: int = 16) -> torch.Tensor:
        """
        Generate latent samples.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Generated latents (num_samples, C, H, W)
        """
        self.model.eval()
        shape = (num_samples,) + tuple(self.latent_shape)
        return self.diffusion.sample(self.model, shape, progress=True)
