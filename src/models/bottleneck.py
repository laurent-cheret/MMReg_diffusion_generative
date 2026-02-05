"""
Bottleneck VAE for Latent Space Compression.

Sits between the SD VAE encoder and decoder to create a semantic
bottleneck from the spatial 4x16x16 latent to a flat 256-d vector.

This gives us:
- Sharp reconstruction (from the frozen SD decoder)
- Stable interpolation (from the tight bottleneck)
- Well-structured latent for diffusion (256-d instead of 1024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BottleneckVAE(nn.Module):
    """
    Small VAE that compresses SD VAE latents (4x16x16) to a flat bottleneck.

    Architecture:
        Encoder: 4x16x16 -> flatten(1024) -> 512 -> mu(latent_dim) + logvar(latent_dim)
        Decoder: latent_dim -> 512 -> 1024 -> reshape(4x16x16)
    """

    def __init__(
        self,
        spatial_shape: tuple = (4, 16, 16),
        latent_dim: int = 256,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.latent_dim = latent_dim
        self.flat_dim = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]

        # Encoder: spatial latent -> bottleneck
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: bottleneck -> spatial latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.flat_dim),
        )

    def encode(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        """
        Encode spatial latent to bottleneck.

        Args:
            x: SD VAE latent (B, 4, 16, 16)
            sample: Whether to sample from posterior or use mean

        Returns:
            Dict with 'z', 'mu', 'logvar'
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if sample and self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        return {'z': z, 'mu': mu, 'logvar': logvar}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode bottleneck back to spatial latent.

        Args:
            z: Bottleneck vector (B, latent_dim)

        Returns:
            Reconstructed spatial latent (B, 4, 16, 16)
        """
        h = self.decoder(z)
        return h.view(-1, *self.spatial_shape)

    def forward(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: SD VAE latent (B, 4, 16, 16)

        Returns:
            Dict with 'x_recon', 'z', 'mu', 'logvar'
        """
        enc = self.encode(x, sample=sample)
        x_recon = self.decode(enc['z'])

        return {
            'x_recon': x_recon,
            'z': enc['z'],
            'mu': enc['mu'],
            'logvar': enc['logvar']
        }


class BottleneckLoss(nn.Module):
    """
    Loss for training the bottleneck VAE.

    total = recon_loss + beta * kl_loss + lambda_mm * mm_loss

    recon_loss: MSE between original and reconstructed SD latent
    kl_loss: KL divergence of bottleneck posterior
    mm_loss: MM-Reg on the bottleneck z
    """

    def __init__(
        self,
        lambda_mm: float = 1.0,
        beta: float = 0.001,
        mm_variant: str = 'correlation'
    ):
        super().__init__()
        self.lambda_mm = lambda_mm
        self.beta = beta

        from .losses import MMRegLoss
        self.mm_loss_fn = MMRegLoss(variant=mm_variant)

    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        r: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            x: Original SD latent (B, 4, 16, 16)
            x_recon: Reconstructed SD latent (B, 4, 16, 16)
            z: Bottleneck vector (B, latent_dim)
            r: Reference PCA embeddings (B, D_ref)
            mu: Posterior mean (B, latent_dim)
            logvar: Posterior log variance (B, latent_dim)

        Returns:
            Dict with loss components
        """
        # Reconstruction loss on SD latents
        recon_loss = F.mse_loss(x_recon, x)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # MM-Reg loss
        mm_loss = self.mm_loss_fn(z, r)

        total_loss = recon_loss + self.beta * kl_loss + self.lambda_mm * mm_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mm_loss': mm_loss
        }
