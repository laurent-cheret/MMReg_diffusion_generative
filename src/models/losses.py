"""
MM-Reg Loss: Manifold-Matching Regularization for VAEs

Core loss functions that enforce pairwise distance preservation between
latent space and reference space (DINOv2 or PCA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances for a batch of vectors.

    Args:
        x: Tensor of shape (B, D) where B is batch size, D is feature dim

    Returns:
        Distance matrix of shape (B, B)
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    dot_product = torch.mm(x, x.t())
    square_norm = dot_product.diag()
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0)  # numerical stability
    return torch.sqrt(distances + 1e-8)


def get_upper_triangular(matrix: torch.Tensor) -> torch.Tensor:
    """Extract upper triangular elements (excluding diagonal) as a flat vector."""
    n = matrix.shape[0]
    indices = torch.triu_indices(n, n, offset=1, device=matrix.device)
    return matrix[indices[0], indices[1]]


class MMRegLoss(nn.Module):
    """
    Manifold-Matching Regularization Loss.

    Enforces that pairwise distances in latent space match those in reference space.
    Supports two variants:
    - 'correlation': Pearson correlation (scale-invariant, recommended)
    - 'si_mse': Scale-invariant MSE with Huber loss
    """

    def __init__(self, variant: str = 'correlation', huber_delta: float = 1.0):
        """
        Args:
            variant: 'correlation' or 'si_mse'
            huber_delta: Delta parameter for Huber loss (only used with si_mse)
        """
        super().__init__()
        self.variant = variant
        self.huber_delta = huber_delta

    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Compute MM-Reg loss between latent vectors and reference vectors.

        Args:
            z: Latent vectors (B, D_latent) - flattened VAE latents
            r: Reference vectors (B, D_ref) - DINOv2 or PCA embeddings

        Returns:
            Scalar loss value
        """
        # Compute pairwise distance matrices
        D_z = pairwise_distances(z)
        D_r = pairwise_distances(r)

        # Extract upper triangular (unique pairs)
        d_z = get_upper_triangular(D_z)
        d_r = get_upper_triangular(D_r)

        if self.variant == 'correlation':
            return self._correlation_loss(d_z, d_r)
        elif self.variant == 'si_mse':
            return self._si_mse_loss(d_z, d_r)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def _correlation_loss(self, d_z: torch.Tensor, d_r: torch.Tensor) -> torch.Tensor:
        """
        Pearson correlation loss: L = 1 - correlation(d_z, d_r)

        This is scale and shift invariant.
        """
        # Center the vectors
        d_z_centered = d_z - d_z.mean()
        d_r_centered = d_r - d_r.mean()

        # Compute correlation
        cov = (d_z_centered * d_r_centered).mean()
        std_z = d_z_centered.std() + 1e-8
        std_r = d_r_centered.std() + 1e-8

        correlation = cov / (std_z * std_r)

        return 1.0 - correlation

    def _si_mse_loss(self, d_z: torch.Tensor, d_r: torch.Tensor) -> torch.Tensor:
        """
        Scale-Invariant MSE with Huber loss.

        Normalizes distances by their means before computing loss.
        """
        # Normalize by mean (detach z normalization to avoid trivial solution)
        d_z_norm = d_z / (d_z.mean().detach() + 1e-8)
        d_r_norm = d_r / (d_r.mean() + 1e-8)

        # Huber loss for robustness
        return F.smooth_l1_loss(d_z_norm, d_r_norm, beta=self.huber_delta)


class VAELoss(nn.Module):
    """
    Combined VAE loss with MM-Reg regularization.

    Total loss = reconstruction_loss + beta * kl_loss + lambda_mm * mm_loss
    """

    def __init__(
        self,
        lambda_mm: float = 0.1,
        beta: float = 1.0,
        mm_variant: str = 'correlation',
        reconstruction_type: str = 'mse'
    ):
        """
        Args:
            lambda_mm: Weight for MM-Reg loss
            beta: Weight for KL divergence (beta-VAE)
            mm_variant: 'correlation' or 'si_mse'
            reconstruction_type: 'mse' or 'lpips'
        """
        super().__init__()
        self.lambda_mm = lambda_mm
        self.beta = beta
        self.reconstruction_type = reconstruction_type
        self.mm_loss = MMRegLoss(variant=mm_variant)

        # LPIPS for perceptual loss (optional)
        self._lpips = None

    @property
    def lpips(self):
        """Lazy load LPIPS to avoid import issues."""
        if self._lpips is None and self.reconstruction_type == 'lpips':
            import lpips
            self._lpips = lpips.LPIPS(net='alex').eval()
            for p in self._lpips.parameters():
                p.requires_grad = False
        return self._lpips

    def reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        if self.reconstruction_type == 'mse':
            return F.mse_loss(x_recon, x)
        elif self.reconstruction_type == 'lpips':
            # LPIPS expects inputs in [-1, 1]
            lpips_model = self.lpips.to(x.device)
            return lpips_model(x, x_recon).mean()
        else:
            raise ValueError(f"Unknown reconstruction type: {self.reconstruction_type}")

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence loss for VAE.

        For diffusers AutoencoderKL, this is computed from the posterior.
        """
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        r: torch.Tensor,
        posterior=None
    ) -> dict:
        """
        Compute combined loss.

        Args:
            x: Original images (B, C, H, W)
            x_recon: Reconstructed images (B, C, H, W)
            z: Latent vectors, flattened (B, D_latent)
            r: Reference vectors (B, D_ref)
            posterior: VAE posterior for KL computation (diffusers style)

        Returns:
            Dictionary with total loss and individual components
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(x, x_recon)

        # KL loss
        if posterior is not None:
            # For diffusers AutoencoderKL
            kl_loss = posterior.kl().mean()
        else:
            kl_loss = torch.tensor(0.0, device=x.device)

        # MM-Reg loss
        mm_loss = self.mm_loss(z, r)

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + self.lambda_mm * mm_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mm_loss': mm_loss
        }
