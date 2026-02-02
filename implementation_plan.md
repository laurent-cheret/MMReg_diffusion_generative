# MM-Reg: Manifold-Matching Regularization for Latent Generative Models (v3)

## 1. Executive Summary & Vision

**The Goal**: Publish a top-tier paper (ICLR/CVPR) demonstrating that preserving *pairwise manifold geometry* in the VAE latent space significantly improves downstream diffusion model training speed and generation quality (FID).

**The Core Hypothesis**: Current VAEs compress images individually, ignoring the relational structure (topology/geometry) of the dataset. By enforcing that the *pairwise distances* in the latent space match the pairwise distances in a robust reference space (DINOv2 or PCA), we create a "flatter", more navigable latent manifold.

**Key Differentiator**: Unlike REPA (diffusion regularization), EQ-VAE (symmetry), or VA-VAE (point-wise alignment), MM-Reg enforces **isometric/topological structural preservation**.

---

## 2. Technical Specification

### 2.1 The MM-Reg Loss Formulation
We define the regularization term to be added to the standard VAE objective.

**Inputs**:
- $Z \in \mathbb{R}^{B \times D_{flat}}$: Batch of flattened latent vectors.
- $R \in \mathbb{R}^{B \times D_{ref}}$: Batch of reference vectors (frozen).

**Distance Computation**:
Let $D^Z$ and $D^R$ be the $B \times B$ pairwise Euclidean distance matrices.
$$D^Z_{ij} = ||z_i - z_j||_2, \quad D^R_{ij} = ||r_i - r_j||_2$$

**Normalization (Crucial for Stability)**:
Naïve distance matching is unstable due to scale differences. We propose **Rank-Correlation** or ** Normalized MSE**.

*Variant A: Scale-Invariant MSE (Si-MSE)*
$$ \hat{D}^Z = \frac{D^Z}{\text{detach}(\text{mean}(D^Z)) + \epsilon}, \quad \hat{D}^R = \frac{D^R}{\text{mean}(D^R) + \epsilon} $$
$$ \mathcal{L}_{\text{MM}} = \text{HuberLoss}(\text{UpperTri}(\hat{D}^Z), \text{UpperTri}(\hat{D}^R)) $$
*Note: Using Huber/SmoothL1 instead of MSE protects against outliers.*

*Variant B: Pearson Correlation (The "MMAE" Approach)*
$$ \mathcal{L}_{\text{MM}} = 1 - \frac{\text{Cov}(D^Z_{upper}, D^R_{upper})}{\sigma(D^Z_{upper})\sigma(D^R_{upper})} $$
*Recommendation*: Start with **Variant B** for robustness to scaling/shifting, but ablation Variant A.

### 2.2 Reference Spaces
1.  **DINOv2-ViT-B/14 (CLS token)**:
    -   *Pros*: Captures high-level semantic distances. "A dog is closer to a cat than a car."
    -   *Cons*: Low spatial resolution. Might encourage semantic clustering at the cost of texture.
2.  **Pixel-PCA (Top $k$ components)**:
    -   *Pros*: Captures dominant *physical* variations (lighting, color, gross shape). Theory-aligned with MMAE.
    -   *Implementation*: Run PCA on a subset of ImageNet (e.g., 50k images) once to get projection matrix $W$. $r_i = W^T x_i$.
3.  **Hybrid (Ablation)**:
    -   $L = \lambda_1 L_{DINO} + \lambda_2 L_{PCA}$.

### 2.3 Base Architecture
-   **Model**: `stabilityai/sd-vae-ft-mse` (Standard SD1.5 VAE, refined).
    -   *Why*: Proven baseline. Training SDXL VAE is computationally heavier (larger channels), though possible if results are promising.
-   **Resolution**: $256 \times 256$ (Latent: $32 \times 32 \times 4$).

---

## 3. Validation Strategy (The "Go/No-Go" Pipeline)

### Stage A: Geometric Validation (The "MMAE" Check)
*Goal: Prove we changed the geometry without breaking reconstruction.*
*Resource: 1 GPU, ~12 hours.*

1.  **Train**: VAE + MM-Reg (5 epochs).
2.  **Metric 1: Distance Correlation**: Compute Pearson correlation between Latent Distances and Reference Distances on validation set.
    -   *Target*: $> 0.8$ (Baseline usually $< 0.5$).
3.  **Metric 2: Linear Probing**: Train a simple linear classifier on frozen latents to predict ImageNet classes.
    -   *Hypothesis*: MM-Reg latents are more linearly separable (higher accuracy).
4.  **Metric 3: Reconstruction**: rFID, PSNR, SSIM.
    -   *Constraint*: Must stay within 1-2% of baseline.

### Stage B: Interpolation Quality (The "Smoothness" Check)
*Goal: Prove the manifold is less "twisted".*
1.  **Perceptual Path Length (PPL)**: Sample latent path $z_t = \text{lerp}(z_1, z_2, t)$. Decode frames.
    -   Measure LPIPS between steps $t$ and $t+\epsilon$.
    -   *Result*: Lower variance in LPIPS steps = smoother manifold.
2.  **Qualitative**: Visualization of grids.

### Stage C: Generation (The "Paper" Results)
*Goal: Prove easier learning for Diffusion.*
*Resource: 1-4 GPUs, 2-4 days.*
1.  **Model**: DiT-S/2 (Small) or SiT-S/2.
2.  **Training**: Train on MM-Reg latents vs Baseline latents.
3.  **Key Metric**: FID @ 50k, 100k, 200k steps.
    -   *Win Condition*: MM-Reg reaches Baseline's final FID 20-30% faster, or achieves lower final FID.

---

## 4. Implementation Details

### 4.1 Project Structure
```
mm-reg/
├── configs/                # Hydra or OmegaConf configs
│   ├── vae/
│   │   ├── mmreg_dinov2.yaml
│   │   └── baseline.yaml
│   └── dit/
│       ├── sit_small.yaml
│       └── sit_xl.yaml
├── src/
│   ├── data/              # ImageNet loader, potential pre-computing references
│   ├── models/
│   │   ├── vae_wrapper.py # Wraps AutoencoderKL to add loss
│   │   ├── losses.py      # The MM-Reg logic
│   │   └── reference.py   # DINOv2 / PCA caching
│   ├── analysis/          # Geometry metrics (intrinsic dim, correlation)
│   └── trainer.py         # Modified training loop
├── scripts/
│   ├── train_vae.py
│   ├── train_dit.py
│   ├── cache_references.py # Pre-compute DINO embeddings for speed
│   └── viz_latents.py
└── tests/                 # Critical for loss correctness
```

### 4.2 Critical Engineering Optimizations
1.  **Reference Caching**: Do NOT run DINOv2 forward pass every iteration during VAE training if input augmentation is simple (e.g. CenterCrop).
    -   *Correction*: VAE training requires Data Augmentation (RandomResizedCrop). We cannot pre-cache perfectly. We must run DINOv2 online *or* use "Weak Augmentation" for Reference and "Strong" for VAE?
    -   *Decision*: Run DINOv2 online in `no_grad`. It's fast (ViT-B) compared to VAE backward pass.
2.  **Gradient Checkpointing**: Enable on VAE Encoder to maximize Batch Size.
    -   *Why*: Pairwise loss quality scales with Batch Size $B$. We need $B \ge 64$.
3.  **Mixed Precision (FP16/BF16)**: Mandatory.

---

## 5. Development Roadmap

1.  **Day 1: The Loss & The Baseline**
    -   Implement `MMRegLoss` (test with random tensors).
    -   Set up `train_vae.py` using `diffusers` AutoencoderKL.
    -   Dry run on "Imagenette" (subset) to verify loss decreases.

2.  **Day 2: VAE Experiments (Stage A)**
    -   Launch Baseline VAE finetune.
    -   Launch MM-Reg VAE finetune.
    -   Implement `evaluate_geometry.py`.

3.  **Day 3: Analysis & Tuning**
    -   Compare Distance Correlation vs Reconstruction tradeoff.
    -   Tune $\lambda_{MM}$.

4.  **Day 4+: Generation (Stage B/C)**
    -   Once VAE is frozen, cache all latents to disk (huge speedup for DiT training).
    -   Train DiT.
