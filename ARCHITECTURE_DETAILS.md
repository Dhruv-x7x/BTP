# DETAILED ARCHITECTURE SPECIFICATIONS

## MiSiCNet Architecture Deep Dive

```
Input: noise_tensor (1, bands+depth, rows, cols) + normalized HSI
       └── Concatenated: original HSI or network input

Forward Pass:
┌─────────────────────────────────────────────────────────────────┐
│ Input tensor shape: (1, depth, rows, cols)                      │
└──────────────┬──────────────────────────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
   SKIP PATH         MAIN PATH
   ────────         ────────
   ↓                ↓
1×1 Conv       3×3 Conv
in→4            in→256
   │             ↓
   │         BatchNorm2d
   │         eps=1e-5
   │             ↓
   │         LeakyReLU
   │         negative_slope=0.1
   │             ↓
   │         3×3 Conv
   │         256→256
   │             ↓
   │         BatchNorm2d
   │         LeakyReLU
   │             ↓
   └─────→ Concatenate (256+4=260)
          ↓
   3×3 Conv (260→256)
   BatchNorm + LeakyReLU
          ↓
   1×1 Conv (256→P) ← ENCODING LAYER
          ↓
    x2 = endmembers (shape: 1, P, 1, 1)
          ↓
   Deconvolution Layer
   (P→P over spatial dims)
          ↓
    x3 = abundances (rows, cols, P)
```

**Key Parameters**:
- Kernel sizes: 3×3 (main conv), 1×1 (skip/encoding), deconv kernels
- Activation: LeakyReLU(0.1) 
- Batch normalization: momentum=0.1, eps=1e-5
- Skip connection width: 4 channels
- Padding: configurable ('zero' or 'reflect')

**Output**:
- Endmembers: shape (P, bands) after reshaping x2
- Abundances: shape (rows, cols, P)

## Vision Transformer (TransNet) Architecture

```
Input: HSI data (bands,) per pixel or (rows*cols, bands)

PATCH EMBEDDING:
├─ Divide into patches: (image_h/patch) × (image_w/patch) patches
├─ Project each patch: (patch²×bands) → dim
└─ Add position embeddings + class token

ViT BACKBONE (N transformer layers):
├─ Self-Attention head
│  ├─ Linear proj: x → Q, K, V
│  ├─ Scaled dot-product: attention = softmax(Q·K^T/√d)
│  ├─ Apply attention: output = attention·V
│  └─ Multi-head: repeat with different W matrices
│
├─ Cross-Attention (between patches and endmembers):
│  └─ Query: patch tokens
│     Key/Value: endmember tokens
│     → learns to focus on relevant spectral information
│
├─ Feed-Forward (per-token MLP):
│  ├─ Linear: dim → hidden_dim
│  ├─ GELU activation
│  ├─ Dropout
│  ├─ Linear: hidden_dim → dim
│  └─ Dropout
│
├─ Residual connections around each block
└─ Layer normalization before each component

DECODER (AutoEncoder):
├─ Fully connected: dim → hidden_dim
├─ Batch norm + ReLU
└─ Fully connected: hidden_dim → P (endmembers)
    └─ Softmax normalization (abundance constraint)

Output: (P,) abundances per pixel
```

**Key Components**:
- `PreNorm`: Layer normalization wrapper
- `FeedForward`: 2-layer MLP with GELU
- `CrossAttention`: Multi-head attention between sequences
- `CrossAttentionBlock`: Attention + residual + norm
- `Transformer`: Stack of layers with cross-attention
- `ViT`: Full Vision Transformer with patch embedding

**Hyperparameters** (from code):
- Patch size: configurable
- Hidden dimension: 768 (typical)
- Number of heads: 8 (typical)
- Depth: typically 4-6 transformer blocks
- Dropout: 0.1-0.2 (typical)

## Loss Functions in Detail

### Reconstruction Loss
```python
L_recon = MSE(Y_true, Y_reconstructed)
        = MSE(Y_true, A @ E)
        where A = abundances, E = endmembers
```

### Sparse Regularization (OSP-based)
```python
L_sparse = ||E @ E^T||_upper  (off-diagonal sum of Gram matrix)
Discourages correlated endmembers
Weighted by λ hyperparameter
```

### Sum-to-One Constraint Loss
```python
L_sum = ||sum(A, axis=1) - 1||_1
Ensures abundances sum to 1 per pixel
L1 norm (absolute error) - robust to outliers
Weighted by γ hyperparameter
```

### Spectral Angle Distance Loss
```python
SAD(v1, v2) = arccos(v1·v2 / (||v1||·||v2||))
Range: 0 to π radians (0 to 180 degrees)
Measures spectral similarity
```

### Total Training Loss
```python
L_total = α·L_recon + λ·L_sparse + γ·L_sum
```

## NFINDR Algorithm Details

```
Input: Y (bands, pixels), p (number of endmembers)

Step 1: Dimensionality Reduction
├─ Compute SVD of (Y - mean(Y))
└─ Project to (p-1) dimensions using PCA

Step 2: Initialization
├─ Random simplex in (p-1) dimensional space
└─ Project back to original space

Step 3: Greedy Simplex Volume Maximization
For max_iter iterations:
  ├─ For each sample in Y:
  │  ├─ Replace each endmember with sample
  │  └─ Compute resulting simplex volume
  └─ Keep configuration with maximum volume

Step 4: Multi-restart Strategy
├─ Run algorithm n_restarts times with different initializations
└─ Return best result

Volume Computation:
├─ For simplex spanned by columns of A (d × p):
│  └─ V = |det(A)| / (p-1)!
└─ Maximizing volume → finding extreme points
```

**Advantages**:
- No training required
- Deterministic (given seed)
- Works well with standard datasets
- Simple and interpretable

## PMM (Probabilistic Mixture Model) Graphical Model

```
Bayesian Generative Model:

Prior over Endmembers:
  M ~ Dir(α)  where α_i = 1/bands
  Shape: (P, bands)
  Interpretation: Each endmember is simplex on spectra

Prior over Abundances (per-pixel):
  A[pixel] ~ Dir(β)  where β_i = 1/P
  Shape: (P,) normalized
  Interpretation: Abundances are simplex

Likelihood (observation model):
  Y[pixel, band] ~ N(A[pixel]^T · M[:, band], σ²)
  Where:
    A[pixel] = abundance vector for pixel
    M[:, band] = spectral values at band
    σ² = noise variance (learned)

Joint Distribution:
  p(Y, M, A, σ²) = p(Y|M,A,σ²) · p(M) · p(A) · p(σ²)

Inference:
  Posterior: p(M, A, σ²|Y)  (intractable)
  
  SVI approximation:
    q(M, A, σ²) ≈ p(M, A, σ²|Y)
    Using:
    - AutoDiagonalNormal (fully factorized)
    - AutoLowRankMultivariateNormal (partial correlations)
  
  MCMC exact:
    NUTS sampler with HMC for geometry
    Produces posterior samples
```

## Evaluation Metrics Formal Definitions

### Spectral Angle Distance (SAD)
```
SAD(v₁, v₂) = arccos(|v₁·v₂| / (||v₁|| · ||v₂||))
Range: [0, π/2] radians (0 to 90 degrees)
Lower is better
```

### Root Mean Square Error (RMSE)
```
RMSE = √(1/N Σᵢ(ŷᵢ - yᵢ)²)
Computed per-endmember on abundance maps
Lower is better
```

### Signal-to-Noise Ratio (SNR) per-endmember
```
SNR_dB = 10·log₁₀(P_signal / P_noise)
     = 10·log₁₀(||A_true||² / ||A_est - A_true||²)
Computed independently per endmember
Higher is better
```

### Reconstruction Error (RE)
```
RE = ||Y - Ŷ||_F / ||Y||_F
Where Ŷ = A_est @ E_est
Frobenius norm (sum of squared elements)
Lower is better
```

### Spatial Entropy
```
For each endmember map A[:,:,i]:
  Entropy_i = -Σₓ,ᵧ p(x,y)·log(p(x,y))
  where p normalized to [0,1]
Measures spatial concentration
Higher entropy → more spread (less sharp)
```

## Training Workflow Pseudocode

```python
for run in range(num_runs):
    # Set random seed
    np.random.seed(seed[run])
    torch.manual_seed(seed[run])
    
    # Load data
    hsi = load_HSI(dataset_path)
    Y_true = hsi.array()  # (pixels, bands)
    
    # Initialize model
    if method == 'MiSiCNet':
        net = MiSiCNet(...)
        net_input = get_noise(...)
    elif method == 'TransNet':
        net = ViT(...)
        net_input = Y_true  # batch processing
    
    # Optimizer
    optimizer = Adam(net.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            E_est, A_est = net(batch)
            
            # Compute loss
            loss_recon = MSE(Y_true, A_est @ E_est)
            loss_sparse = sparse_loss(E_est)
            loss_constraint = sum_to_one_loss(A_est)
            
            loss_total = loss_recon + λ·loss_sparse + γ·loss_constraint
            
            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Metrics & Visualization
        if epoch % save_interval == 0:
            E_est_np = E_est.detach().cpu().numpy()
            A_est_np = A_est.detach().cpu().numpy()
            
            # Reorder to match GT
            idx_E = order_endmembers(E_est_np, E_true)
            idx_A = order_abundance(A_est_np, A_true)
            
            # Compute metrics
            sad = compute_SAD(E_est_np[idx_E], E_true)
            rmse = compute_RMSE(A_est_np[..., idx_A], A_true)
            snr = compute_SNR_per_endmember(hsi, A_est_np, E_est_np)
            
            # Save results
            save_visualizations(E_est_np, A_est_np, output_dir)
            save_csv_results(sad, rmse, output_dir)
```

