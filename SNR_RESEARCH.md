# SNR (Signal-to-Noise Ratio) Research for Hyperspectral Unmixing

## What is SNR in Hyperspectral Context?

### Definition
```
SNR (dB) = 10 * log10(Signal_Power / Noise_Power)
```

### Common Formulations

**1. Image-level SNR (Overall)**
```
SNR = 10 * log10(||Y||² / ||N||²)
```
Where:
- Y = clean signal (ground truth or estimated)
- N = noise

**2. Per-Band SNR**
```
SNR_band_i = 10 * log10(mean(Y[:, i])² / var(N[:, i]))
```
- Each spectral band has different SNR
- Useful for identifying noisy bands

**3. Per-Pixel SNR**
```
SNR_pixel_j = 10 * log10(||Y[j,:]||² / ||N[j,:]||²)
```
- SNR varies spatially across image

**4. Reconstruction SNR**
```
SNR_recon = 10 * log10(||Y||² / ||Y - Ŷ||²)
```
Where:
- Y = original image
- Ŷ = reconstructed image (from unmixing: Ŷ = A · E)

---

## Before/After SNR in Unmixing Context

### Scenario 1: Denoising Quality

**Before:** SNR of noisy observed image
```python
# Assume we have ground truth clean signal Y_clean
SNR_before = 10 * np.log10(np.linalg.norm(Y_clean)**2 /
                            np.linalg.norm(Y_observed - Y_clean)**2)
```

**After:** SNR of reconstructed/denoised image
```python
# After unmixing, we get reconstruction Ŷ = A · E
Y_reconstructed = A @ E
SNR_after = 10 * np.log10(np.linalg.norm(Y_clean)**2 /
                          np.linalg.norm(Y_reconstructed - Y_clean)**2)
```

**Interpretation:**
- SNR_after > SNR_before → Unmixing **improved** signal quality (denoised)
- SNR_after < SNR_before → Unmixing **degraded** signal quality

---

### Scenario 2: Per-Endmember SNR

**Before:** SNR of raw mixed pixels for each endmember
```python
# For endmember k, identify pixels where it's dominant
dominant_pixels_k = np.where(A_groundtruth[:,:,k] > 0.5)

# Compute SNR for those pixels
signal_k = Y_clean[dominant_pixels_k]  # Clean signal
noise_k = Y_observed[dominant_pixels_k] - Y_clean[dominant_pixels_k]  # Noise

SNR_before_k = 10 * np.log10(np.mean(signal_k**2) / np.mean(noise_k**2))
```

**After:** SNR after unmixing extracts endmember k
```python
# Use extracted endmember E_est[k] and abundances A_est
reconstructed_k = A_est[:,:,k][:,:,np.newaxis] * E_est[k]

# Compute SNR for endmember k's contribution
signal_k_true = A_groundtruth[:,:,k][:,:,np.newaxis] * E_groundtruth[k]
error_k = reconstructed_k - signal_k_true

SNR_after_k = 10 * np.log10(np.linalg.norm(signal_k_true)**2 /
                            np.linalg.norm(error_k)**2)
```

**Interpretation:**
- Shows how well unmixing recovered each individual endmember's signal

---

### Scenario 3: Abundance SNR

**Before:** SNR of noisy abundance observations (if we had direct abundance measurements)
- Usually not applicable, since we don't observe abundances directly

**After:** SNR of estimated abundances vs ground truth
```python
# Per endmember
for k in range(P):
    A_true_k = A_groundtruth[:,:,k]
    A_est_k = A_estimated[:,:,k]
    error_k = A_est_k - A_true_k

    SNR_abundance_k = 10 * np.log10(np.mean(A_true_k**2) /
                                     np.mean(error_k**2))
```

---

## Research Findings from Literature

### Key Papers

**1. "Noise Reduction in Hyperspectral Images Through Spectral Unmixing"**
- SNR varies band-by-band
- Low SNR bands need stronger denoising
- High SNR bands preserve details
- **Before/After:** Compares SNR of original noisy image vs denoised image

**2. "Performance Evaluation of Various Hyperspectral Nonlinear Unmixing Algorithms"**
- Tests algorithms under different SNR conditions: SNR = 10 dB, 15 dB, 20 dB
- Uses **average SAD** for endmember accuracy
- Uses **RMSE** for abundance accuracy
- Shows that PISINMF has best performance at SNR = 15 dB

**3. "Phase-Locked SNR Band Selection for Weak Mineral Signal Detection" (2025)**
- Proposes SNR-guided band selection
- Three stages:
  1. Phase-locked spectral smoothing + SNR-guided band selection
  2. Unsupervised endmember extraction + NNLS unmixing
  3. Abundance map generation
- **Key insight:** Select high-SNR bands for better endmember extraction

**4. PNAS Study on Raman Spectroscopy**
- Acquired high SNR (1,920 spectra, 5s integration) and low SNR (7,680 spectra, 0.5s integration)
- Systematic evaluation of unmixing under different SNR conditions
- **Before/After:** Compares high SNR vs low SNR acquisition

---

## What Should We Measure?

### Option A: Reconstruction SNR (Most Common)
**Definition:** How well does reconstructed image Y_recon = A·E match original Y?

```python
def compute_reconstruction_snr(Y_original, E_estimated, A_estimated):
    """
    Compute SNR of reconstructed image
    """
    Y_recon = A_estimated @ E_estimated  # Reconstruction

    signal_power = np.linalg.norm(Y_original.flatten())**2
    noise_power = np.linalg.norm((Y_original - Y_recon).flatten())**2

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db
```

**Before:** Assume original image has some noise level
- Could synthesize: Add Gaussian noise to clean image, measure SNR
- `Y_noisy = Y_clean + noise`
- `SNR_before = 10 * log10(||Y_clean||² / ||noise||²)`

**After:** Measure reconstruction quality
- `Y_recon = A_est · E_est`
- `SNR_after = 10 * log10(||Y_original||² / ||Y_original - Y_recon||²)`

---

### Option B: Per-Endmember SNR (From Current Code)

Looking at the existing code `compute_snr_per_endmember()`, it likely computes:

```python
def compute_snr_per_endmember(hsi, out_avg_np, Eest):
    """
    hsi: Original HSI object
    out_avg_np: Reconstructed image (?)
    Eest: Estimated endmembers
    """
    # Need to examine exact implementation
    # Likely computes SNR for each endmember's contribution
```

**What it should do:**
For each endmember k:
1. Get ground truth abundance A_gt[:,:,k] and endmember E_gt[k]
2. Compute ground truth contribution: `signal_k = A_gt[:,:,k] * E_gt[k]`
3. Get estimated contribution: `recon_k = A_est[:,:,k] * E_est[k]`
4. Compute SNR: `SNR_k = 10 * log10(||signal_k||² / ||signal_k - recon_k||²)`

---

### Option C: Wavelet-Based SNR (Advanced)

**Why wavelets?**
- Separate signal into different frequency components
- Noise often concentrated in high frequencies
- Signal (endmembers) often in low frequencies

**How it works:**
```python
import pywt

def wavelet_snr(signal_1d, wavelet='db4', level=3):
    """
    Compute SNR at different wavelet scales
    """
    # Decompose signal
    coeffs = pywt.wavedec(signal_1d, wavelet, level=level)

    # coeffs[0] = approximation (low freq, signal)
    # coeffs[1:] = details (high freq, noise)

    signal_power = np.sum(coeffs[0]**2)
    noise_power = np.sum([np.sum(c**2) for c in coeffs[1:]])

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db
```

**Before/After with Wavelets:**
- **Before:** Wavelet SNR of original mixed pixel spectra
- **After:** Wavelet SNR of reconstructed spectra from unmixing

**Advantage:** More robust SNR estimate, separates signal from noise in frequency domain

---

## Recommendation for Your Implementation

### Proposed Approach: Reconstruction SNR + Per-Endmember SNR

**Step 1: Overall Reconstruction SNR**
```python
def compute_overall_snr(Y_original, A_est, E_est):
    """
    Overall reconstruction quality
    """
    Y_recon = np.einsum('ijk,kl->ijl', A_est, E_est)  # or reshape + matmul

    mse = np.mean((Y_original - Y_recon)**2)
    signal_power = np.mean(Y_original**2)

    snr_db = 10 * np.log10(signal_power / mse)
    return snr_db
```

**Step 2: Per-Endmember SNR**
```python
def compute_per_endmember_snr(A_gt, E_gt, A_est, E_est):
    """
    SNR for each endmember's contribution
    """
    P = E_gt.shape[0]  # Number of endmembers
    snr_per_endmember = np.zeros(P)

    for k in range(P):
        # Ground truth contribution of endmember k
        # A_gt shape: (rows, cols, P), E_gt shape: (P, bands)
        signal_k = A_gt[:,:,k][:,:,np.newaxis] * E_gt[k]  # (rows, cols, bands)

        # Estimated contribution
        recon_k = A_est[:,:,k][:,:,np.newaxis] * E_est[k]

        # SNR
        signal_power = np.sum(signal_k**2)
        error_power = np.sum((signal_k - recon_k)**2)

        snr_per_endmember[k] = 10 * np.log10(signal_power / (error_power + 1e-10))

    return snr_per_endmember
```

**Step 3: Before/After Comparison**

**"Before"** = Assume original image has known noise level
```python
# If you have clean ground truth Y_clean and noisy observations Y_noisy:
noise = Y_noisy - Y_clean
SNR_before = 10 * np.log10(np.sum(Y_clean**2) / np.sum(noise**2))
```

If you don't have ground truth clean image:
```python
# Use spectral unmixing as denoising
# "Before" = noisy input
# "After" = reconstructed (hopefully denoised) image
Y_input = hsi.data  # Observed (noisy) image
Y_recon = A_est @ E_est  # Reconstructed image

# Assume ground truth is approximately the reconstruction
# (this is circular, but common in practice)
SNR_before = compute_snr_with_noise_estimate(Y_input)
SNR_after = compute_overall_snr(Y_input, A_est, E_est)
```

---

## What's Wrong with Current Implementation?

Need to examine `compute_snr_per_endmember()` in the notebook:

**Potential issues:**
1. **Undefined "before":** What baseline is SNR compared against?
2. **Wrong formula:** Might be computing something other than SNR
3. **Missing ground truth:** SNR needs reference (clean signal or true abundances)
4. **Per-band vs per-endmember confusion:** Are we measuring SNR of spectral bands or endmember contributions?

---

## Action Items

1. ✅ **Examine current `compute_snr_per_endmember()` implementation**
2. **Define clear "before" and "after":**
   - Before: Input image SNR (if noise level known) OR per-endmember signal quality before unmixing
   - After: Reconstructed image SNR OR per-endmember signal quality after unmixing
3. **Implement corrected SNR function:**
   - Overall reconstruction SNR
   - Per-endmember SNR
   - Optional: Wavelet-based SNR for robustness
4. **Save SNR metrics:**
   - CSV: `SNR_before`, `SNR_after`, `SNR_improvement`
   - Per endmember: `SNR_endmember_k` for k=1..P
5. **Visualize:**
   - Bar plot: SNR before vs after
   - Per-endmember SNR comparison across models

---

## Questions for User

1. **Do you have clean ground truth images?**
   - If yes: We can compute true SNR as ||Y_clean||² / ||Y_noisy - Y_clean||²
   - If no: We use reconstruction error as proxy

2. **What should "before" represent?**
   - Option A: SNR of input image (requires noise estimate)
   - Option B: Baseline SNR using simple method (e.g., NFINDR)
   - Option C: SNR without unmixing (just raw pixel spectra)

3. **Should we use wavelets?**
   - Wavelets give more robust SNR estimates
   - But add complexity and computation time

4. **Per-endmember or overall SNR?**
   - Both? (probably yes)
   - Which is more important for your analysis?

---

## Next Step

Let me examine the current `compute_snr_per_endmember()` code to see what it's actually doing, then propose a fix.
