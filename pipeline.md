The notebook implements, trains, and evaluates four distinct hyperspectral image unmixing algorithms:
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

## ALL IMPORTS AND DEPENDENCIES

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

## FUNCTION DEFINITIONS

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

## CLASS DEFINITIONS

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


## DATA PROCESSING STEPS AND WORKFLOWS

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

## DATA VISUALIZATION AND PLOTTING

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
