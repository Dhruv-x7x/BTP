# COMPREHENSIVE AUDIT: am101_dataset_research.ipynb

## 1. OVERALL PURPOSE AND STRUCTURE

**Purpose**: This notebook implements, trains, and evaluates four distinct hyperspectral image unmixing algorithms:
- **MiSiCNet**: Deep learning CNN-based approach
- **TransNet**: Vision Transformer-based approach  
- **NFINDR**: Classical N-Findr endmember extraction algorithm
- **PMM**: Probabilistic Mixture Model using Pyro (variational inference)

**Framework**: Google Colab-based research notebook for comparing multiple spectral unmixing techniques on hyperspectral datasets (Samson, Apex, Urban, Jasper)

**Total Cells**: 36 cells (2 markdown headers, 34 code/content)

**Organization**:
1. Imports & Configuration (Cells 0-4)
2. Utility Functions (Cells 5-11)
3. Model Definitions (Cells 15-23)
4. Training Classes (Cell 25)
5. Experimental Runs (Cells 26-35)

---

## 2. ALL IMPORTS AND DEPENDENCIES

### Core Scientific Libraries
- `numpy` - Numerical computations
- `scipy` (linalg, optimize, io as sio) - Linear algebra, optimization, MATLAB file I/O
- `torch`, `torch.nn` - PyTorch deep learning framework
- `torch.optim` - Optimization algorithms
- `torchvision.transforms` - Image transformations

### Visualization
- `matplotlib.pyplot` - Plotting
- `matplotlib` - Matplotlib core
- `mpl_toolkits.axes_grid1.make_axes_locatable` - Advanced axis handling
- `PIL.Image` - Image file handling

### Deep Learning & Model Components
- `timm.layers.DropPath` - Drop path regularization from timm library
- `timm.models.vision_transformer.Mlp` - MLP from timm Vision Transformers
- `einops`, `einops.layers.torch.Rearrange` - Tensor reshaping operations
- `h5py` - HDF5 file I/O

### Machine Learning & Data
- `sklearn.decomposition.PCA` - Principal Component Analysis
- `pandas` - Data frame manipulation
- `psutil` - System resource monitoring
- `glob` - File pattern matching

### Probabilistic Programming
- `pyro` - Probabilistic programming framework
- `pyro.distributions` (dist) - Probabilistic distributions
- `pyro.infer` - Inference methods (SVI, MCMC, NUTS, Predictive, Trace_ELBO, TraceMeanField_ELBO)
- `pyro.infer.autoguide` - AutoDiagonalNormal, AutoLowRankMultivariateNormal

### Utilities
- `time` - Timing operations
- `os` - Operating system functions
- `random` - Random operations
- `math.factorial` - Mathematical operations
- `IPython.display.display` - Jupyter output display
- `scipy.optimize.minimize` - Optimization
- `scipy.optimize.nnls` - Non-negative least squares

---

## 3. FUNCTION DEFINITIONS (108 TOTAL)

### Core Utility Functions

#### Noise Generation & Initialization
- `OSP(B, R)` - Orthogonal Subspace Projection; computes ||B @ B.T||_upper for endmember orthogonality
- `fill_noise(x, noise_type, *, mean=0.0, std=1.0, low=0.0, high=1.0)` - Fills tensor with uniform or normal noise
- `get_noise(input_depth, method, spatial_size, noise_type='n', var=1./10)` - Generates random tensor for network initialization (noise or meshgrid)

### Network Architecture & Optimization
- `get_params(opt_over, net, net_input, downsampler=None)` - Extracts parameters for optimization over specified components
- `conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride')` - Wrapper for convolutional layers with flexible padding

### Distance & Extraction Metrics
- `Eucli_dist(x, y)` - Euclidean distance between vectors x and y
- `Endmember_extract(x, p)` - Greedy endmember extraction algorithm; finds p endmembers from data matrix x (D x N)

### Data I/O
- `load_HSI(path)` - Loads hyperspectral image from .mat or HDF5 file; returns HSI object with normalized data, dimensions, and ground truth endmembers/abundances

### Performance Metrics
- `numpy_RMSE(y_true, y_pred)` - Root Mean Square Error computation
- `numpy_SAD(y_true, y_pred)` - Spectral Angle Distance between spectral vectors
- `order_abundance(abundance, abundanceGT)` - Reorders estimated abundance maps to best match ground truth (Hungarian algorithm-like matching)
- `order_endmembers(endmembers, endmembersGT)` - Reorders estimated endmembers to match ground truth using SAD
- `compute_abundances_fcls(Y, E, rows=None, cols=None, verbose=False)` - Computes abundance maps using Fully Constrained Least Squares (FCLS) optimization

### Visualization Functions
- `plotEndmembersAndGT(endmembers, endmembersGT, endmember_path, sadsave, save)` - Displays estimated vs ground truth endmembers with SAD scores
- `plotAbundancesSimple(abundances, abundanceGT, abundance_path, rmsesave)` - Displays abundance maps for each endmember with RMSE overlay
- `plotAbundancesGT(abundanceGT, abundance_path)` - Displays ground truth abundance maps
- `plotAbundancesAndGT(abundances, abundanceGT, abundance_path)` - Side-by-side estimated vs GT abundance visualization

### Dimensionality Reduction & Endmember Detection
- `pca(X, d)` - PCA projection; reduces (L, N) data to (L, d) dimensional subspace
- `hyperVca(M, q)` - Hyperspectral Virtual Dimensionality (HyperVCA) for automatic endmember count detection

### Loss Functions
- `Nuclear_norm(inputs)` - Computes nuclear norm (sum of singular values) of input tensor
- `compute_sad(inp, target)` - Spectral Angle Distance loss computation between input and target spectral signatures

### Advanced Diagnostic Metrics
- `compute_snr_per_endmember(hsi, out_avg_np, Eest)` - Computes SNR (Signal-to-Noise Ratio) for each endmember independently; useful for assessing unmixing quality per-component
- `compute_entropy(abundance_maps)` - Computes spatial entropy of abundance maps to assess unmixing sharpness

### Post-Processing & Reporting
- `report_time_and_memory(start_time)` - Prints elapsed time and peak memory usage
- `stack_and_stats(list_of_arrays)` - Aggregates results across multiple runs, computing means and standard deviations
- Various internal hooks (`_make_hook`, `hook`, `train_step`, `my_loss`) for training diagnostics

### Helper Functions in Execution Section
- `fcls_pixel(y, M, tol=1e-8)` - Computes FCLS abundance for single pixel using constrained optimization
- `safe_load_csv(p)` - Safely loads CSV, handling missing files and non-numeric data
- `mean_std_str(arr)` - Extracts mean and std from array
- `latex_table_endmembers(means, stds, metric_name, label)` - Generates LaTeX table from per-endmember statistics
- `show_if_exists(path, title=None, maxsize=(900,900))` - Conditionally displays image if file exists
- `summarize_csv(path, name)` - Loads CSV and prints mean/std across runs
- `show_image_if_exists(path, title=None)` - Displays image using PIL/matplotlib

---

## 4. CLASS DEFINITIONS (22 TOTAL)

### Data Handling Classes

#### HSI - Hyperspectral Image Container
```python
class HSI:
    def __init__(self, data, rows, cols, gt, abundance_gt):
        # Stores normalized HSI data in (rows, cols, bands) format
        # Stores ground truth endmembers (gt) and abundances (abundance_gt)
    
    def array(self):
        # Returns flattened array (rows*cols, bands) for processing
```

#### TrainData - PyTorch Dataset for Training
```python
class TrainData(torch.utils.data.Dataset):
    def __init__(self, img, transform=None)
    def __getitem__(self, index) -> returns single pixel tensor
    def __len__(self) -> returns total number of pixels
```

#### Data - Multi-dataset Manager
```python
class Data:
    def __init__(self, dataset, device)
    def get(self, typ) -> returns HSI object for dataset
    def get_loader(self, batch_size) -> returns PyTorch DataLoader
```

#### NonZeroClipper - Custom Gradient Clipper
```python
class NonZeroClipper(object):
    def __call__(self, module) -> clips gradients to [-1, 1] for non-zero weights
```

### Loss Functions

#### SparseKLloss (nn.Module)
- Computes sparse KL loss using nuclear norm regularization
- `__call__(inp, decay)` - Returns scaled nuclear norm loss

#### SumToOneLoss (nn.Module)  
- Enforces abundance sum-to-one constraint
- `__call__(inp, gamma_reg)` - Returns L1 loss between summed abundances and 1

#### SAD (nn.Module)
- Spectral Angle Distance loss
- `forward(inp, target)` - Computes cosine distance between spectral signatures

#### SID (nn.Module)
- Spectral Information Divergence loss
- `forward(inp, target)` - Computes KL-divergence based spectral divergence

### Deep Learning Model Architectures

#### MiSiCNet (nn.Module) - Convolutional Network
```
Architecture:
- Input: HSI (rows, cols, bands)
- Skip connection path: 1x1 conv → 256 → 256 → 4 channels
- Main path: 3x3 conv → BN → LeakyReLU (repeated)
- Concatenation of skip and main paths
- Upsampling layers
- Encoding layer (1x1 conv to P channels)
- Deconvolutional abundance reconstruction
Output: 
  - x2: Endmember matrix (1, P, 1, 1) 
  - x3: Abundance map (rows, cols, P)
```

#### Vision Transformer Components
- `PreNorm(nn.Module)` - Layer normalization wrapper around any function
- `FeedForward(nn.Module)` - MLP with GELU and dropout
- `CrossAttention(nn.Module)` - Multi-head cross-attention between two sequences
- `CrossAttentionBlock(nn.Module)` - Cross-attention + residual connection
- `Transformer(nn.Module)` - Stack of transformer layers with cross-attention
- `ViT(nn.Module)` - Full Vision Transformer with:
  - Patch embedding (divides image into patches)
  - Positional embeddings
  - Class token
  - Multiple transformer blocks
  - Classification head

#### AutoEncoder (nn.Module)
- Encoder: Linear → BN → ReLU
- Decoder: Linear → softmax normalization
- Used in TransNet for abundance reconstruction

### Classical Algorithms

#### NFINDRModel - N-Findr Endmember Extraction
```python
class NFINDRModel:
    def __init__(self, p, n_restarts=10, max_iter=5, seed=None, use_pca=True)
    def fit(self, Y) -> extracts endmembers using greedy simplex volume maximization
    def fit_transform(self, Y) -> returns endmembers and abundance maps
```
- Uses PCA dimensionality reduction
- Greedy endmember replacement strategy
- Computes simplex volume to maximize unmixing

#### NFINDRRunner - Orchestrator for NFINDR
```python
class NFINDRRunner:
    def __init__(self, dataset, DATASET_PATH, OUTPUT_PATH, device, n_restarts, max_iter)
    def run(self, runs=1, seeds=None) -> executes multiple runs, saves results, produces diagnostic visualizations
```

### Probabilistic Inference

#### PMMRunner - Probabilistic Mixture Model
```python
class PMMRunner:
    def __init__(self, dataset='Samson', DATASET_PATH='.', OUTPUT_PATH='Results',
                 device=None, inference='svi', svi_steps=2000, svi_lr=5e-3,
                 mcmc_samples=500, mcmc_warmup=200, use_cholesky_prior=False,
                 prefer_vi_for_mcmc=False, predictive_samples=50, save_samples=False, seed=1)
    
    def _pyro_model(self, Y, K, device) -> Bayesian graphical model with:
        - Dirichlet prior on endmembers M: Dir(α) where α = 1/bands
        - Dirichlet prior on abundances A: Dir(β) where β = 1/P
        - Normal likelihood with learned precision
    
    def _fit_svi(self, Y_torch, K, steps, lr) -> Stochastic Variational Inference
    def _fit_mcmc(self, Y_torch, K, num_samples, warmup_steps) -> MCMC with NUTS sampler
    
    def run(self, runs=1, seeds=None) -> executes multiple runs, produces CSV summaries
```

### Training Orchestrators

#### MiSiCTrainer - MiSiCNet Training Manager
```python
class MiSiCTrainer:
    def __init__(self, dataset='Samson', DATASET_PATH='.', OUTPUT_PATH='Results', device='cuda')
    
    def run(self, runs=1, seedrng=1, epochs_no=600):
        - Loads HSI dataset
        - Initializes MiSiCNet with network input (noise tensor)
        - Defines loss: reconstruction + sparsity + sum-to-one constraints
        - Trains with Adam optimizer
        - Saves endmembers, abundances per run
        - Computes per-endmember SNR, entropy, RMSE metrics
        - Saves activation visualizations (read-only hooks)
        - Produces CSV results: SAD, abundance RMSE, reconstruction error
```

#### TransNetTrainer - Vision Transformer Training Manager
```python
class TransNetTrainer:
    def __init__(self, dataset='Samson', DATASET_PATH='.', OUTPUT_PATH='Results', device='cuda')
    
    def run(self, num_runs=1, seeds=None, epochs_override=None, save_training_visuals_every=100):
        - Uses ViT + AutoEncoder architecture
        - Batch-wise training on flattened pixels
        - Saves per-epoch visualizations (abundances, reconstruction errors)
        - Saves activation AND attention map visualizations
        - Produces same CSV metrics as MiSiCNet
```

---

## 5. DATA PROCESSING STEPS AND WORKFLOWS

### Data Loading Pipeline
1. `load_HSI(path)` → reads .mat or HDF5 file
2. Normalization: divide by max value across all bands
3. Create HSI object storing:
   - Reshaped image: (rows, cols, bands)
   - Ground truth endmembers: (P, bands)
   - Ground truth abundances: (rows, cols, P)

### Endmember Initialization Strategies
1. **Network Input (MiSiCNet/TransNet)**: Random tensor (noise or meshgrid) concatenated with PCA-reduced HSI
2. **Extraction-based**: Use `Endmember_extract()` or `pca()` + `hyperVca()` for initialization
3. **NFINDR**: Greedy simplex volume maximization with PCA reduction
4. **PMM**: Learned via Bayesian inference

### Training Loop (Deep Models)
```
for epoch in range(epochs):
    loss = reconstruction_loss(Y_recon, Y_true) 
         + λ_sparse * sparse_loss(A)  # OSP or sparse KL
         + γ_reg * sum_to_one_loss(A)
    
    optimizer.step(loss)
    
    if save_interval:
        compute_metrics(A_est, E_est, A_gt, E_gt)
        save_visualization()
```

### Metric Computation Workflow
For each run:
1. Reorder estimated endmembers to match GT using SAD ordering
2. Reorder abundance maps using MSE-based matching
3. For each endmember:
   - SAD(E_est, E_gt)
   - RMSE(A_est, A_gt)
   - SNR per endmember
   - Spatial entropy of abundance
4. Reconstruction error: ||Y - A_est @ E_est||
5. Aggregate across runs: mean ± std

### Post-Processing
1. Load CSV results from all runs
2. Compute aggregated statistics (mean, std per endmember)
3. Generate LaTeX tables for publication
4. Save activation/attention visualizations as PNG
5. Display results in Jupyter notebook

---

## 6. MODELS, ALGORITHMS, AND ANALYSIS TECHNIQUES

### Deep Learning Approaches

#### MiSiCNet (Mixed-resolution Siamese CNN)
- **Type**: Fully CNN-based end-to-end learning
- **Input**: Network input tensor (random) + HSI features
- **Output**: Endmembers E (P x bands) + Abundances A (rows x cols x P)
- **Architecture**:
  - Skip connection: 1x1 conv for fine features
  - Main path: Multi-layer 3x3 convolutions
  - Concatenation and upsampling
  - Encoding to endmember space
  - Deconvolution for abundance maps
- **Training Objective**:
  - L2 reconstruction: ||Y - A·E||²
  - Sparse regularization: λ·OSP(E) or Nuclear norm(A)
  - Abundance constraint: γ·||sum(A,1) - 1||₁
- **Advantages**: Direct end-to-end optimization, no explicit initialization needed

#### TransNet (Vision Transformer-based)
- **Type**: Attention-based unmixing network
- **Architecture**:
  - Patch embedding: divide image into patches, linear projection
  - Positional embedding: learnable position encodings
  - Transformer blocks: Multi-head self-attention + cross-attention
  - AutoEncoder decoder: Projects to endmember space, softmax normalization
- **Training**: Batch-wise pixel processing, same loss as MiSiCNet
- **Advantages**: Global receptive field via attention, can capture long-range dependencies

### Classical Spectral Unmixing Algorithms

#### NFINDR (N-Findr)
- **Type**: Geometric endmember extraction
- **Method**:
  1. Reduce data to (p-1) dimensional subspace using PCA
  2. Initialize with random simplex in subspace
  3. Iteratively replace vertices that minimize simplex volume
  4. Multi-restart strategy for robustness
- **Output**: Endmembers in original spectral space
- **Advantages**: Model-free, no training required, deterministic
- **Abundance Calculation**: FCLS (Fully Constrained Least Squares)
  - Minimizes ||y - E·a||² subject to: a ≥ 0, sum(a) = 1
  - Solved via constrained quadratic programming

#### Probabilistic Mixture Model (PMM)
- **Type**: Bayesian generative model
- **Graphical Model**:
  ```
  M ~ Dir(α)           # Endmembers (P × bands)
  A[i,j] ~ Dir(β)      # Abundances per pixel (P,)
  Y[i,j,k] ~ N(A[i,j]·M[k], σ²)  # Observation model
  ```
- **Priors**:
  - Endmembers: Dirichlet(α=1/bands) - symmetric
  - Abundances: Dirichlet(β=1/P) - symmetric
  - Variance: Gamma prior (optional)
- **Inference Options**:
  - **SVI (Stochastic Variational Inference)**:
    - Variational families: AutoDiagonalNormal or AutoLowRankMultivariateNormal
    - Mini-batch optimization with Adam
    - Scales to large datasets
  - **MCMC (Markov Chain Monte Carlo)**:
    - NUTS (No-U-Turn Sampler) for HMC
    - Produces posterior samples
    - More computationally expensive but asymptotically exact
- **Advantages**: Principled Bayesian approach, uncertainty quantification

### Evaluation Metrics

#### Spectral Angle Distance (SAD)
- Measures angle between two spectral vectors
- SAD(v1, v2) = arccos(v1·v2 / (||v1||·||v2||))
- Per-endmember SAD for quality assessment

#### Root Mean Square Error (RMSE)
- Measures abundance reconstruction error
- RMSE = √(mean((A_est - A_gt)²))
- Computed per-endmember and averaged

#### Signal-to-Noise Ratio (SNR)
- Ratio of signal power to noise power
- Computed independently per endmember
- Measured before and after unmixing

#### Spectral Information Divergence (SID)
- Kullback-Leibler divergence between normalized spectra
- Treats spectra as probability distributions

#### Reconstruction Error (RE)
- RE = ||Y - A_est @ E_est||_F / ||Y||_F
- Measures overall fidelity of unmixing

#### Spatial Entropy
- Measures sharpness of abundance maps
- Higher entropy → more spread out abundances
- Can indicate over/under-regularization

---

## 7. DATA VISUALIZATION AND PLOTTING

### Visualization Functions

1. **Endmember Comparison**:
   - `plotEndmembersAndGT()`: Side-by-side plots of estimated vs ground truth
   - Shows per-endmember SAD scores as titles
   - Uses spectral signature line plots

2. **Abundance Maps**:
   - `plotAbundancesSimple()`: Grid of spatial abundance maps (one per endmember)
   - Color bar shows intensity ranges
   - RMSE values overlaid as text
   - `plotAbundancesAndGT()`: Estimated vs GT side-by-side

3. **Activation Visualizations**:
   - MiSiCTrainer/TransNetTrainer: Saves activation feature maps
   - Shows first 8 channels of selected layers
   - Helps diagnose network learning

4. **Attention Visualizations** (TransNet):
   - Saves self-attention maps from transformer blocks
   - Shows what regions network focuses on
   - Per-head attention weight matrices

5. **Training Progress**:
   - Per-epoch reconstruction error plots
   - Per-epoch abundance RMSE tracking
   - Exported as PNG every N epochs

### Result Aggregation & Reporting

1. **CSV Outputs**:
   - `{dataset}_endmember_SAD_results.csv`: Per-endmember SAD across runs
   - `{dataset}_abundance_RMSE_results.csv`: Per-endmember RMSE across runs
   - `{dataset}_reconstruction_errors.csv`: Overall reconstruction error per run
   - `{dataset}_SNR_before_after.csv`: SNR metrics per endmember

2. **Summary Statistics**:
   - Computes mean ± std across multiple runs
   - Generates LaTeX tables for publication

3. **Directory Structure**:
   ```
   OUTPUT_PATH/
   ├── MiSiCNet/{dataset}/
   │   ├── run{i}/
   │   │   ├── E_run{i}.npy  # Endmembers
   │   │   ├── A_run{i}.npy  # Abundances
   │   │   └── {dataset}_run{i}.mat  # MATLAB format
   │   ├── endmember/
   │   │   └── {dataset}_run{i}.png
   │   ├── abundance/
   │   │   └── {dataset}_run{i}.png
   │   ├── activations/
   │   │   ├── *_activation.png
   │   │   └── *_channel_stats.csv
   │   └── {dataset}_*_results.csv
   ├── TAEU/{dataset}/  # TransNet outputs
   │   └── (similar structure)
   ├── NFINDR/{dataset}/
   └── PMM/{dataset}/
   ```

---

## 8. OVERALL FLOW AND ORGANIZATION

### Notebook Execution Flow

**Phase 1: Setup (Cells 0-4)**
- Title and references
- Import all dependencies
- Define dataset paths

**Phase 2: Utility Functions (Cells 5-11)**
- Noise generation and network initialization
- Euclidean distance and endmember extraction
- HSI data container class
- Performance metric computation
- Visualization functions

**Phase 3: Model Definitions (Cells 15-23)**
- Loss functions: SparseKL, SumToOne, SAD, SID
- MiSiCNet CNN architecture
- Vision Transformer components (PreNorm, Attention, ViT, AutoEncoder)
- NFINDR classical algorithm
- PMM Bayesian probabilistic model

**Phase 4: Training Classes (Cell 25)**
- MiSiCTrainer: Orchestrates MiSiCNet training with diagnostics
- TransNetTrainer: Orchestrates Vision Transformer training
- Diagnostic helpers for SNR, entropy, activation visualization

**Phase 5: Experimental Execution (Cells 26-35)**
- Cell 28-29: Run MiSiCNet with multiple seeds, aggregate results
- Cell 30-31: Run TransNet with multiple seeds, generate LaTeX tables
- Cell 32-33: Run NFINDR algorithm
- Cell 34-35: Run PMM with SVI/MCMC inference

### Key Design Patterns

1. **Modular Training**: Separate Trainer classes for each algorithm
2. **Multi-run Experiments**: All methods support seed-based reproducibility
3. **Automatic Result Aggregation**: CSV export and statistical summarization
4. **Diagnostic Collection**: Per-method activation/attention visualization
5. **Flexible Dataset Support**: Samson, Apex, Urban, Jasper datasets
6. **GPU/CPU Support**: All algorithms configurable for device selection

### Dependencies Between Components

```
load_HSI 
├── returns HSI object
│   ├── used by MiSiCTrainer
│   ├── used by TransNetTrainer
│   ├── used by NFINDRRunner
│   └── used by PMMRunner
│
order_endmembers + order_abundance
├── used by diagnostic reporting
└── used by metric visualization
│
compute_abundances_fcls
├── used by NFINDR (abundance computation)
└── used by PMM (initialization and reporting)
│
plotEndmembersAndGT + plotAbundancesSimple
└── used by all trainers for result visualization
```

### Configuration Management
- Dataset selection: 'Samson', 'Apex', 'Urban', 'Jasper'
- Number of runs per method: typically 5 (with seeds [101, 202, 303, 404, 505])
- Device selection: 'cuda' or 'cpu'
- Model-specific hyperparameters:
  - MiSiCNet: epochs, learning rate (implicit in optimizer)
  - TransNet: epochs, learning rate
  - NFINDR: n_restarts, max_iter
  - PMM: svi_steps, svi_lr, mcmc_samples, inference method

---

## SUMMARY STATISTICS

- **Total Functions**: 108
- **Total Classes**: 22
- **Total Code Cells**: 34
- **Total Notebook Size**: ~50K tokens (very large)
- **Algorithms Implemented**: 4 major (MiSiCNet, TransNet, NFINDR, PMM)
- **Datasets Supported**: 4 (Samson, Apex, Urban, Jasper)
- **Evaluation Metrics**: 6+ (SAD, RMSE, SNR, RE, Entropy, SID)
- **Multi-run Experiments**: Standard 5-run protocol with seed variation
- **Output Formats**: .mat files, PNG images, CSV statistics, LaTeX tables

