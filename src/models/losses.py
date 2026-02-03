"""
MM-Reg Loss: Manifold-Matching Regularization for VAEs

Core loss functions that enforce pairwise distance preservation between
latent space and pre-computed PCA reference space.

Key insight: We pre-compute PCA projections for ALL training samples.
During training, each batch looks up its corresponding PCA embeddings
and compares pairwise distances: samples far apart in PCA space should
remain far apart in latent space.
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
        super().__init__()
        self.variant = variant
        self.huber_delta = huber_delta

    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Compute MM-Reg loss between latent vectors and reference vectors.

        Args:
            z: Latent vectors (B, D_latent) - flattened VAE latents
            r: Reference vectors (B, D_ref) - pre-computed PCA projections

        Returns:
            Scalar loss value (0 = perfect correlation, 2 = perfect anti-correlation)
        """
        # Force FP32 to avoid overflow in pairwise distance computation
        # (FP16 can overflow with large vectors like 4096-dim latents)
        z = z.float()
        r = r.float()

        # Compute pairwise distance matrices
        D_z = pairwise_distances(z)
        D_r = pairwise_distances(r)

        # Extract upper triangular (unique pairs, excludes diagonal)
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

        This is scale and shift invariant - dimensionality of PCA doesn't
        need to match latent space.
        """
        d_z_centered = d_z - d_z.mean()
        d_r_centered = d_r - d_r.mean()

        cov = (d_z_centered * d_r_centered).mean()
        std_z = d_z_centered.std() + 1e-8
        std_r = d_r_centered.std() + 1e-8

        correlation = cov / (std_z * std_r)
        return 1.0 - correlation

    def _si_mse_loss(self, d_z: torch.Tensor, d_r: torch.Tensor) -> torch.Tensor:
        """Scale-Invariant MSE with Huber loss."""
        d_z_norm = d_z / (d_z.mean().detach() + 1e-8)
        d_r_norm = d_r / (d_r.mean() + 1e-8)
        return F.smooth_l1_loss(d_z_norm, d_r_norm, beta=self.huber_delta)


class VAELoss(nn.Module):
    """
    Combined VAE loss with MM-Reg regularization.

    Total loss = reconstruction_loss + beta * kl_loss + lambda_mm * mm_loss

    IMPORTANT scaling notes:
    - recon_loss (MSE): typically ~0.01 for good reconstruction
    - kl_loss: HUGE (~70,000) because it's summed over 4x32x32=4096 latent dims
    - mm_loss: 0-2 range (1 - correlation)

    Default beta=1e-6 scales KL to ~0.07, comparable to recon and mm.
    """

    def __init__(
        self,
        lambda_mm: float = 1.0,
        beta: float = 1e-6,  # Very small! KL is ~70k unscaled
        mm_variant: str = 'correlation',
        reconstruction_type: str = 'mse'
    ):
        """
        Args:
            lambda_mm: Weight for MM-Reg loss (default 1.0)
            beta: Weight for KL divergence (default 1e-6, KL is naturally ~70k)
            mm_variant: 'correlation' or 'si_mse'
            reconstruction_type: 'mse' or 'lpips'
        """
        super().__init__()
        self.lambda_mm = lambda_mm
        self.beta = beta
        self.reconstruction_type = reconstruction_type
        self.mm_loss_fn = MMRegLoss(variant=mm_variant)
        self._lpips = None

    @property
    def lpips(self):
        if self._lpips is None and self.reconstruction_type == 'lpips':
            import lpips
            self._lpips = lpips.LPIPS(net='alex').eval()
            for p in self._lpips.parameters():
                p.requires_grad = False
        return self._lpips

    def reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        if self.reconstruction_type == 'mse':
            return F.mse_loss(x_recon, x)
        elif self.reconstruction_type == 'lpips':
            lpips_model = self.lpips.to(x.device)
            return lpips_model(x, x_recon).mean()
        else:
            raise ValueError(f"Unknown reconstruction type: {self.reconstruction_type}")

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
            r: Reference vectors (B, D_ref) - PRE-COMPUTED PCA projections
            posterior: VAE posterior for KL computation

        Returns:
            Dictionary with total loss and individual components
        """
        recon_loss = self.reconstruction_loss(x, x_recon)

        if posterior is not None:
            kl_loss = posterior.kl().mean()  # Mean over batch, sum over dims
        else:
            kl_loss = torch.tensor(0.0, device=x.device)

        mm_loss = self.mm_loss_fn(z, r)

        # Properly scaled total loss
        total_loss = recon_loss + self.beta * kl_loss + self.lambda_mm * mm_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mm_loss': mm_loss
        }
