"""
Training logic for MM-Reg VAE.

Handles:
- Training loop with mixed precision
- Logging and checkpointing
- Evaluation metrics
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import os
from tqdm import tqdm
import json
from pathlib import Path


class MMRegTrainer:
    """
    Trainer for MM-Reg VAE finetuning.
    """

    def __init__(
        self,
        vae: nn.Module,
        reference_model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda',
        use_amp: bool = True,
        log_interval: int = 50,
        save_dir: str = './checkpoints',
        scheduler: Optional[Any] = None
    ):
        """
        Args:
            vae: MMRegVAE model
            reference_model: Reference extractor (DINOv2/PCA)
            loss_fn: VAELoss with MM-Reg
            optimizer: Optimizer
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device
            use_amp: Use automatic mixed precision
            log_interval: Steps between logging
            save_dir: Directory for checkpoints
            scheduler: Learning rate scheduler
        """
        self.vae = vae
        self.reference_model = reference_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.scheduler = scheduler

        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Logging
        self.train_history = []
        self.val_history = []

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of average metrics
        """
        self.vae.train()
        epoch_metrics = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'mm_loss': 0.0
        }
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # Get reference embeddings (no grad)
            with torch.no_grad():
                ref_features = self.reference_model(images)

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.vae(images)
                    losses = self.loss_fn(
                        x=images,
                        x_recon=outputs['x_recon'],
                        z=outputs['latent_flat'],
                        r=ref_features,
                        posterior=outputs['posterior']
                    )

                # Backward with scaling
                self.scaler.scale(losses['loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.vae(images)
                losses = self.loss_fn(
                    x=images,
                    x_recon=outputs['x_recon'],
                    z=outputs['latent_flat'],
                    r=ref_features,
                    posterior=outputs['posterior']
                )

                losses['loss'].backward()
                self.optimizer.step()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key] += losses[key].item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{losses['loss'].item():.4f}",
                    'recon': f"{losses['recon_loss'].item():.4f}",
                    'mm': f"{losses['mm_loss'].item():.4f}"
                })

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of average metrics
        """
        if self.val_loader is None:
            return {}

        self.vae.eval()
        val_metrics = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'mm_loss': 0.0
        }
        num_batches = 0

        for images, _ in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)

            # Get reference embeddings
            ref_features = self.reference_model(images)

            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.vae(images, sample=False)  # Use mean for val
                    losses = self.loss_fn(
                        x=images,
                        x_recon=outputs['x_recon'],
                        z=outputs['latent_flat'],
                        r=ref_features,
                        posterior=outputs['posterior']
                    )
            else:
                outputs = self.vae(images, sample=False)
                losses = self.loss_fn(
                    x=images,
                    x_recon=outputs['x_recon'],
                    z=outputs['latent_flat'],
                    r=ref_features,
                    posterior=outputs['posterior']
                )

            # Accumulate metrics
            for key in val_metrics:
                val_metrics[key] += losses[key].item()
            num_batches += 1

        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches

        return val_metrics

    def train(self, num_epochs: int, save_every: int = 1):
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}, AMP: {self.use_amp}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            print(f"\nEpoch {epoch} - Train: " +
                  f"loss={train_metrics['loss']:.4f}, " +
                  f"recon={train_metrics['recon_loss']:.4f}, " +
                  f"mm={train_metrics['mm_loss']:.4f}")

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)

                print(f"Epoch {epoch} - Val: " +
                      f"loss={val_metrics['loss']:.4f}, " +
                      f"recon={val_metrics['recon_loss']:.4f}, " +
                      f"mm={val_metrics['mm_loss']:.4f}")

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best.pt')

            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')

        # Save final checkpoint
        self.save_checkpoint('final.pt')
        self.save_history()

        print("Training complete!")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'vae_state_dict': self.vae.state_dict(),
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
        print(f"Saved checkpoint: {self.save_dir / filename}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)

        self.vae.load_state_dict(checkpoint['vae_state_dict'])
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
        """Save training history to JSON."""
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
