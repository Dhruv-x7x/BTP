# Fourier + SVM-RBF Research for Hyperspectral Unmixing

## Research Findings from Literature

### Key Finding 1: FFT for Dimensionality Reduction

**Source:** "Application of Spectral Unmixing on Hyperspectral Data" (MDPI)

**Key Points:**
- **FFT-based unmixing is the quickest approach**
- Linear least squares unmixing (LLSU) with Fast Fourier Transform achieves:
  - Efficient dimensionality reduction
  - High unmixing accuracy
- Faster than traditional PCA-based dimensionality reduction

**How it works:**
1. Apply FFT to each pixel's spectrum
2. Keep low-frequency components (dimensionality reduction)
3. Apply unmixing in frequency domain
4. Transform back to spatial domain if needed

---

### Key Finding 2: SVM with RBF Kernel for Classification

**Source:** "Spectral-Similarity-Based Kernel of SVM" (MDPI)

**Key Points:**
- SVM with **spectral-similarity-based kernels** (including RBF) for hyperspectral classification
- **Power-SAM-RBF kernel** provides outstanding performance when:
  - Similarity between spectral signatures is extremely high, OR
  - Similarity is extremely low
- One-against-all multi-class SVM with Gaussian (RBF) kernel applied to abundance-based features

---

### Key Finding 3: SVM for Abundance Estimation

**Source:** "A supervised abundance estimation method for hyperspectral unmixing" (Remote Sensing Letters)

**Key Points:**
- SVM can be used for **abundance estimation** (not just classification!)
- Supervised approach: Train SVM on known abundance examples
- Nonlinear unmixing algorithms more efficient than linear ones for feature extraction
- **Spatial regularization greatly improves classification accuracy**

---

### Key Finding 4: Kernel-Based Nonlinear Unmixing

**Source:** "Reweighted Kernel-Based Nonlinear Hyperspectral Unmixing" (ResearchGate)

**Key Points:**
- Kernel methods (including RBF kernels) map spectra to higher-dimensional feature space
- Nonlinear mixing can be approximated as linear in kernel space
- RBF kernel: `k(x, y) = exp(-γ ||x - y||²)`
- Regional ℓ₁-norm regularization for sparsity

---

## Proposed Pipelines

### Pipeline Option A: FFT → Feature Engineering → SVM-RBF → Abundances

**Architecture:**
```
Input: Pixel spectrum y (L bands)
  ↓
FFT: Compute Fourier transform
  - F(y) = FFT(y)  # L complex coefficients
  ↓
Feature Extraction:
  - Magnitude: |F(y)| (L real values)
  - Keep low frequencies: F_low = |F(y)|[:K]  # K << L (e.g., K=20)
  - Optional: Add phase information
  ↓
Feature Vector: x = F_low (K dimensions)
  ↓
SVM-RBF Regression: Train one SVM per endmember
  - SVM_k(x) → abundance_k  for k = 1..P
  ↓
Post-process: Apply constraints
  - Non-negativity: a = max(a, 0)
  - Sum-to-one: a = a / sum(a)
  ↓
Output: Abundance vector a (P values)
```

**Training Data:**
- For each pixel in training set:
  - Compute FFT features: x = FFT_features(pixel)
  - Get ground truth abundances: a_gt
  - Train pair: (x, a_gt)

**Advantages:**
- FFT provides dimensionality reduction (L → K)
- Low-frequency components capture main spectral structure
- SVM-RBF can learn nonlinear abundance relationships
- Fast inference after training

**Disadvantages:**
- Requires ground truth abundances for training (supervised)
- FFT assumes some periodicity in spectral signatures (may not hold)
- Need to enforce abundance constraints post-hoc

---

### Pipeline Option B: Endmember Extraction + FFT + SVM Abundances

**Architecture:**
```
Stage 1: Endmember Extraction
  - Use NFINDR on original spectra
  - Extract P endmembers: E (P × L)

Stage 2: FFT Feature Extraction
  - For each pixel y:
    - Compute FFT: F(y) = FFT(y)
    - Extract magnitude: m = |F(y)|[:K]  # Low frequencies
    - Feature vector: x = m

  - For each endmember E[p]:
    - Compute FFT: F(E[p]) = FFT(E[p])
    - Extract magnitude: m_p = |F(E[p])|[:K]
    - Endmember features: x_E[p] = m_p

Stage 3: SVM-RBF for Abundance Estimation
  - For each endmember p, train SVM:
    - Input: FFT features x
    - Output: Abundance fraction a_p
    - Kernel: RBF with learned γ

  - Prediction:
    - For pixel x: a_p = SVM_p(x)
    - Apply constraints: a = constrain(a)

Output: Abundances A (rows × cols × P)
```

**Training:**
- If ground truth abundances available:
  - Train supervised: (FFT_features, abundance_gt)
- If no ground truth:
  - Use FCLS abundances as pseudo-labels
  - Train semi-supervised

**Advantages:**
- Separates endmember extraction from abundance estimation
- Can use unsupervised NFINDR for endmembers
- FFT features might capture spectral shape better than raw bands
- SVM-RBF can learn complex spectral-abundance relationships

---

### Pipeline Option C: FFT Denoising + Traditional Unmixing

**Architecture:**
```
Stage 1: FFT-Based Denoising
  - For each pixel y:
    - FFT: F(y) = FFT(y)
    - Low-pass filter: F_filtered = F(y) * H(ω)
      where H(ω) = 1 if ω < ω_cutoff, else 0
    - Inverse FFT: y_clean = IFFT(F_filtered)

Stage 2: Traditional Unmixing
  - NFINDR on y_clean to get E
  - FCLS to get A
  OR
  - Train SVM-RBF on y_clean (no FFT features, just denoised spectra)

Output: Endmembers E, Abundances A
```

**Advantages:**
- Simple: FFT just for denoising
- Works with any unmixing algorithm
- Interpretable (just removes high-frequency noise)

**Disadvantages:**
- Might remove important spectral features (if signal also has high-freq components)
- Choosing cutoff frequency ω_cutoff is tricky

---

## SVM-RBF Implementation Details

### Abundance Constraints Problem

Standard SVM regression doesn't enforce:
1. **Non-negativity:** a_p ≥ 0
2. **Sum-to-one:** Σ a_p = 1

**Solutions:**

**Option 1: Post-processing (Simple)**
```python
from sklearn.svm import SVR

# Train P independent SVMs
svms = []
for p in range(P):
    svm = SVR(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, A_train[:, p])  # X_train = FFT features
    svms.append(svm)

# Predict
A_pred = np.zeros((N_test, P))
for p in range(P):
    A_pred[:, p] = svms[p].predict(X_test)

# Apply constraints
A_pred = np.maximum(A_pred, 0)  # Non-negativity
A_pred = A_pred / A_pred.sum(axis=1, keepdims=True)  # Sum-to-one
```

**Pros:** Simple, easy to implement
**Cons:** Constraints not part of optimization, may hurt accuracy

---

**Option 2: Softmax Transformation (Better)**
```python
# Train SVMs to predict unconstrained logits
logits_pred = np.zeros((N_test, P))
for p in range(P):
    logits_pred[:, p] = svms[p].predict(X_test)

# Apply softmax to get abundances
A_pred = np.exp(logits_pred) / np.exp(logits_pred).sum(axis=1, keepdims=True)
```

**Pros:** Automatically satisfies sum-to-one AND non-negativity
**Cons:** Need to transform ground truth during training:
```python
# During training, transform abundances to logits
A_logits_train = np.log(A_train + 1e-10)  # Inverse of softmax
```

---

**Option 3: Multi-output SVM with Custom Loss (Advanced)**
```python
from sklearn.multioutput import MultiOutputRegressor

# Use constrained optimization
# (requires custom solver, not in scikit-learn by default)
```

**Pros:** Constraints integrated into training
**Cons:** Complex to implement

---

### RBF Kernel Parameter γ

**Definition:** `K(x, y) = exp(-γ ||x - y||²)`

**γ controls:**
- γ small → wide kernel, smooth decision boundary (underfitting)
- γ large → narrow kernel, complex decision boundary (overfitting)

**Tuning γ:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'gamma': [0.001, 0.01, 0.1, 1, 10], 'C': [0.1, 1, 10, 100]}
svm = SVR(kernel='rbf')
grid = GridSearchCV(svm, param_grid, cv=5)
grid.fit(X_train, y_train)

best_gamma = grid.best_params_['gamma']
best_C = grid.best_params_['C']
```

**Default:** `gamma='scale'` in sklearn uses `1 / (n_features * X.var())`

---

## FFT Feature Engineering

### Basic FFT Features

```python
from scipy.fft import fft, fftfreq

def extract_fft_features(spectrum, n_features=20):
    """
    Extract low-frequency FFT features from spectrum

    Args:
        spectrum: (L,) array of spectral values
        n_features: Number of low-frequency components to keep

    Returns:
        features: (n_features,) array of FFT magnitude features
    """
    # Compute FFT
    fft_vals = fft(spectrum)

    # Get magnitude of low frequencies
    fft_mag = np.abs(fft_vals[:n_features])

    return fft_mag
```

### Advanced FFT Features

**Include phase information:**
```python
def extract_fft_features_advanced(spectrum, n_features=20):
    fft_vals = fft(spectrum)[:n_features]

    # Magnitude and phase
    magnitude = np.abs(fft_vals)
    phase = np.angle(fft_vals)

    # Concatenate
    features = np.concatenate([magnitude, phase])
    return features
```

**Frequency binning:**
```python
def extract_fft_features_binned(spectrum, n_bins=10):
    """
    Average FFT magnitudes into frequency bins
    """
    fft_mag = np.abs(fft(spectrum))

    # Divide into bins
    bin_size = len(fft_mag) // (2 * n_bins)  # Only use first half (symmetric)
    features = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        bin_avg = np.mean(fft_mag[start:end])
        features.append(bin_avg)

    return np.array(features)
```

---

## Complete Implementation Plan

### Recommended Pipeline: Option B (NFINDR + FFT + SVM-RBF)

**Why Option B?**
- Endmember extraction is unsupervised (NFINDR) → no need for labeled endmembers
- FFT features for dimensionality reduction → faster than using all L bands
- SVM-RBF for abundance estimation → can be semi-supervised (use FCLS pseudo-labels if no ground truth)
- Faster than deep learning models → satisfies "low-compute baseline" requirement

---

### Step-by-Step Implementation

**Step 1: Prepare Data**
```python
# Load HSI
hsi = load_HSI('path/to/samson.mat')
Y = hsi.array()  # (N_pixels, L_bands)
rows, cols, L = hsi.rows, hsi.cols, Y.shape[1]
N = rows * cols

# Ground truth (if available)
E_gt = hsi.gt  # (P, L)
A_gt = hsi.abundance_gt  # (rows, cols, P)
A_gt_flat = A_gt.reshape(N, -1)  # (N, P)
```

**Step 2: Extract Endmembers (NFINDR)**
```python
from sklearn.decomposition import PCA

# NFINDR to get endmembers
nfindr = NFINDRModel(p=P, n_restarts=10, max_iter=5)
E_est = nfindr.fit(Y.T)  # Input: (L, N), Output: (P, L)
```

**Step 3: Extract FFT Features**
```python
def extract_fft_features_batch(Y, n_features=20):
    """
    Y: (N_pixels, L_bands)
    Returns: (N_pixels, n_features)
    """
    N, L = Y.shape
    X_fft = np.zeros((N, n_features))

    for i in range(N):
        fft_vals = fft(Y[i])
        X_fft[i] = np.abs(fft_vals[:n_features])

    return X_fft

n_fft_features = 20  # Tune this
X_fft = extract_fft_features_batch(Y, n_features=n_fft_features)
```

**Step 4: Get Training Labels**

**If ground truth abundances available:**
```python
X_train = X_fft
y_train = A_gt_flat  # (N, P)
```

**If no ground truth (semi-supervised):**
```python
# Use FCLS to get pseudo-labels
from scipy.optimize import nnls

def fcls_abundance(y_pixel, E):
    """Fully constrained least squares for one pixel"""
    # Solve: min ||y - E.T @ a||^2  s.t. a >= 0, sum(a) = 1
    # Approximate with NNLS + normalization
    a, _ = nnls(E.T, y_pixel)
    a = a / (a.sum() + 1e-10)  # Normalize
    return a

# Get pseudo-labels
y_train = np.zeros((N, P))
for i in range(N):
    y_train[i] = fcls_abundance(Y[i], E_est)

X_train = X_fft
```

**Step 5: Train SVM-RBF (One per Endmember)**
```python
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Split data
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Train one SVM per endmember
svms = []
for p in range(P):
    print(f"Training SVM for endmember {p+1}/{P}")
    svm = SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.01)
    svm.fit(X_tr, y_tr[:, p])
    svms.append(svm)

    # Validation score
    val_pred = svm.predict(X_val)
    val_rmse = np.sqrt(np.mean((val_pred - y_val[:, p])**2))
    print(f"  Validation RMSE: {val_rmse:.4f}")
```

**Step 6: Predict Abundances**
```python
# Predict on all pixels (or test set)
A_pred_flat = np.zeros((N, P))
for p in range(P):
    A_pred_flat[:, p] = svms[p].predict(X_fft)

# Apply constraints
A_pred_flat = np.maximum(A_pred_flat, 0)  # Non-negativity
A_pred_flat = A_pred_flat / A_pred_flat.sum(axis=1, keepdims=True)  # Sum-to-one

# Reshape to image
A_pred = A_pred_flat.reshape(rows, cols, P)
```

**Step 7: Evaluate**
```python
# Compute metrics
from metrics import numpy_RMSE, numpy_SAD, order_abundance, order_endmembers

# Order to match ground truth
A_pred_ordered = order_abundance(A_pred, A_gt)

# RMSE per endmember
rmse_per_em = []
for p in range(P):
    rmse = numpy_RMSE(A_gt[:,:,p], A_pred_ordered[:,:,p])
    rmse_per_em.append(rmse)
    print(f"Endmember {p+1} RMSE: {rmse:.4f}")

# SAD for endmembers
E_ordered = order_endmembers(E_est, E_gt)
sad_per_em = []
for p in range(P):
    sad = numpy_SAD(E_gt[p], E_ordered[p])
    sad_per_em.append(sad)
    print(f"Endmember {p+1} SAD: {sad:.4f}")
```

---

## Expected Performance

### Computational Cost
- **Training time:** ~1-5 minutes (SVM training for P=3-4 endmembers)
- **Inference time:** ~0.1 seconds (FFT + SVM prediction very fast)
- **Memory:** Low (only store P SVMs)

**Much faster than:**
- MiSiCNet: 600 epochs, ~30-60 minutes
- TransNet: 500 epochs, ~30-60 minutes
- PMM: 100-200 SVI steps, ~10-20 minutes

### Accuracy Expectations
- **Likely worse than deep models** (MiSiCNet, TransNet)
- **Possibly better than NFINDR** (if FFT features help)
- **Main advantage:** Speed + interpretability

### Tuning Knobs
1. `n_fft_features`: Number of FFT components (10-50)
2. `C` and `gamma` in SVM: Regularization and kernel width
3. `epsilon` in SVR: Margin for regression
4. Endmember extraction method: NFINDR vs VCA vs others

---

## Action Items

1. ✅ **Research complete:** Understand FFT + SVM approach
2. **Implement pipeline Option B:** NFINDR + FFT + SVM-RBF
3. **Test on Samson:** Verify it works, measure time and metrics
4. **Tune hyperparameters:** n_fft_features, C, gamma
5. **Test on Apex:** Generalization check
6. **Compare with other models:** Add to results table

---

## Questions for User

1. **Do we have ground truth abundances for Samson/Apex?**
   - If yes: Train SVM supervised
   - If no: Use FCLS pseudo-labels

2. **Should FFT be for dimensionality reduction or denoising?**
   - Option B uses it for features
   - Option C uses it for denoising

3. **How many FFT features to use?**
   - Start with 20, can tune (10-50 range)

4. **Should we tune SVM hyperparameters or use defaults?**
   - Tuning will improve accuracy but takes time
   - For MVP, defaults might be enough

---

## Next Step

Ready to implement once you confirm:
1. Which pipeline option? (Recommend Option B)
2. Use ground truth abundances or pseudo-labels?
3. How many FFT features to start with? (Recommend 20)
