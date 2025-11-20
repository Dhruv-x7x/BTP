# Critical Questions & Research Tasks for MVP

## 🔥 HIGH PRIORITY ISSUES

### 1. PMM Performance Problem

**Issue:** PMM takes too long even with SVI on T4 GPU

**Hypotheses:**
- Not using GPU correctly (tensors on CPU?)
- Too many SVI steps
- Inefficient variational guide (AutoLowRankMultivariateNormal too complex?)
- Batch processing not optimal
- Some operations forcing CPU/GPU sync

**Investigation Checklist:**
- [ ] Profile PMM code with `torch.profiler` or `line_profiler`
- [ ] Check device placement: `print(tensor.device)` for all tensors
- [ ] Count SVI steps - is it 2000? Can we reduce to 500-1000?
- [ ] Check guide type - can we use simpler AutoDiagonalNormal?
- [ ] Check if MCMC is being used instead of SVI accidentally
- [ ] Try subsampling pixels (test on 50% of image first)
- [ ] Compare with original PMM paper implementation (if available)

**Questions for User:**
1. How long does PMM currently take on Samson dataset?
2. What specific line of code shows it's using SVI (not MCMC)?
3. Can you share the exact error/warning messages if any?

---

### 2. SNR Metric - What's Wrong?

**Issue:** SNR currently working incorrectly

**Questions to Answer:**
1. **What is the current implementation doing?**
   - Need to examine `compute_snr_per_endmember()` in detail
   - What formula is it using?

2. **What should SNR measure in unmixing?**
   - Option A: Image reconstruction SNR = `10 * log10(||Y||² / ||Y - Ŷ||²)` where Ŷ = A·E
   - Option B: Per-endmember abundance SNR = `10 * log10(||A_true||² / ||A_true - A_est||²)`
   - Option C: Per-endmember spectral SNR for extracted endmembers

3. **Before/After comparison - what are we comparing?**
   - Scenario A: SNR of **raw image** (before unmixing) vs SNR of **reconstructed image** (after unmixing)
   - Scenario B: SNR of **initial endmember estimates** vs **final endmember estimates**
   - Scenario C: Something else?

4. **Where do wavelets come in?**
   - Do we compute SNR at each wavelet scale?
   - Do we use wavelets to **separate signal from noise** first, then compute SNR?
   - Is this for the "before" measurement?

**Research Tasks:**
- [ ] Review hyperspectral unmixing papers for SNR definitions
- [ ] Check how SNR is used in benchmark papers (MiSiCNet, TransNet papers)
- [ ] Understand wavelet-based SNR estimation (if needed)

**Current Code Analysis Needed:**
```python
# From audit, this function exists:
def compute_snr_per_endmember(hsi, out_avg_np, Eest):
    # What does this actually compute?
    # Need to examine line by line
```

---

### 3. Fourier + SVM-RBF Pipeline Design

**Issue:** Need to research and design this baseline from scratch

#### Question Set A: What does Fourier Transform do here?

**Potential Use Cases:**
1. **Feature extraction:** FFT of each pixel's spectrum → use magnitude/phase as features
2. **Denoising:** FFT → zero out high-frequency components → inverse FFT → cleaner spectrum
3. **Dimensionality reduction:** Keep only low-frequency components (natural compression)
4. **Band selection:** Identify important frequency bands

**Most likely for unmixing:** Option 1 or 3 (feature extraction or compression)

**Research Tasks:**
- [ ] Search Google Scholar: "Fourier transform hyperspectral unmixing"
- [ ] Search: "frequency domain hyperspectral analysis"
- [ ] Look for papers combining spectral FFT with machine learning

---

#### Question Set B: How do Fourier and SVM-RBF fit together?

**Proposed Pipeline Options:**

**Option A: End-to-End Abundance Prediction**
```
Input: Pixel spectrum (L bands)
  ↓
FFT: Extract frequency features (L/2 + 1 magnitudes + phases)
  ↓
Feature Engineering: Select K most important frequency components
  ↓
SVM-RBF: Train K regressors (one per endmember) to predict abundances
  ↓
Post-process: Apply sum-to-one + non-negativity constraints
  ↓
Output: Abundance vector (P endmembers)
```

**Endmember extraction:** Use NFINDR or assume ground truth endmembers

---

**Option B: Two-Stage (Endmember Extraction + Abundance Estimation)**
```
Stage 1: Endmember Extraction
  - Use NFINDR or VCA on original spectra
  - Get endmembers E (P × L)

Stage 2: Fourier + SVM for Abundances
  Input: Pixel spectrum y (L,)
    ↓
  FFT: Get frequency features f (K,)
    ↓
  SVM-RBF: Predict abundances a (P,) using features f
    ↓
  Constrain: sum(a) = 1, a ≥ 0
    ↓
  Output: Abundances a (P,)
```

---

**Option C: Fourier as Pre-processing Only**
```
Input: Hyperspectral image Y (rows × cols × L)
  ↓
FFT: Denoise by filtering high frequencies
  ↓
Inverse FFT: Get cleaned image Y_clean
  ↓
Traditional Pipeline: Use NFINDR or FCLS on Y_clean
```
*(This might not need SVM at all)*

---

#### Question Set C: SVM-RBF Specifics

**SVM Configuration Questions:**
1. **One SVM per endmember** (P separate SVMs) or **multi-output SVM**?
   - Separate SVMs: More flexible, can tune per endmember
   - Multi-output: Faster, shares information across endmembers

2. **How to enforce constraints?**
   - **Non-negativity:** a ≥ 0
     - Option 1: Use SVM's built-in constraints (if available)
     - Option 2: Post-processing: `a = np.maximum(a, 0)`
   - **Sum-to-one:** Σa = 1
     - Option 1: Softmax normalization: `a = exp(a) / sum(exp(a))`
     - Option 2: L1 normalization: `a = a / sum(a)` (after non-negativity)

3. **What are the inputs and outputs?**
   - **Input:** Frequency features from FFT (K features)
   - **Output:** Abundances (P values)
   - **Training data:** Pairs (frequency_features, ground_truth_abundances)

4. **Training set construction:**
   - Use all pixels from Samson as training samples?
   - Or train/test split?

**Scikit-learn Implementation:**
```python
from sklearn.svm import SVR  # For regression

# One SVR per endmember
svms = []
for k in range(P):  # P endmembers
    svm_k = SVR(kernel='rbf', C=1.0, gamma='scale')
    svm_k.fit(X_train_freq_features, y_train_abundances[:, k])
    svms.append(svm_k)

# Predict on test data
abundances_pred = np.zeros((N_test, P))
for k in range(P):
    abundances_pred[:, k] = svms[k].predict(X_test_freq_features)

# Apply constraints
abundances_pred = np.maximum(abundances_pred, 0)  # Non-negativity
abundances_pred = abundances_pred / abundances_pred.sum(axis=1, keepdims=True)  # Sum-to-one
```

---

#### Research Tasks for Fourier + SVM-RBF

**Literature Review:**
- [ ] Search: "SVM hyperspectral unmixing"
- [ ] Search: "frequency domain hyperspectral classification"
- [ ] Search: "Fourier features remote sensing"
- [ ] Check if any papers combine FFT + SVM specifically for unmixing

**Similar Baselines in Literature:**
- Look for papers using classical ML (SVM, RF, etc.) for unmixing
- Look for papers using frequency-domain analysis in hyperspectral imaging

**Potential References:**
- SVM for abundance estimation papers
- Kernel methods for hyperspectral unmixing
- Frequency-domain denoising in remote sensing

---

### Design Decision Needed (After Research)

Once we review the literature, we need to decide:

1. **Which pipeline option to implement** (A, B, or C)?
2. **What frequency features to extract** (magnitude only? phase too? how many components?)
3. **How to construct training data** (all pixels? subsample? train/test split?)
4. **How to enforce constraints** (during training? post-processing?)
5. **How to extract endmembers** (use NFINDR? assume GT? learn jointly?)

**Decision Criteria:**
- Simplicity (need to implement by end of week)
- Computational efficiency (must run faster than deep models - that's the point of baseline)
- Interpretability (classical baseline should be interpretable)
- Performance (doesn't need to beat deep models, but should be reasonable)

---

## Summary of Immediate Research Tasks

| Task | Priority | Est. Time | Status |
|------|----------|-----------|--------|
| Profile PMM code to find bottleneck | 🔥 Critical | 2 hours | ⏸️ Pending |
| Examine current SNR implementation | 🔥 Critical | 1 hour | ⏸️ Pending |
| Research SNR definitions in unmixing papers | 🔥 Critical | 2 hours | ⏸️ Pending |
| Literature review: Fourier + hyperspectral | 🔥 Critical | 2 hours | ⏸️ Pending |
| Literature review: SVM + unmixing | 🔥 Critical | 2 hours | ⏸️ Pending |
| Design Fourier+SVM pipeline | 🔥 Critical | 1 hour | ⏸️ Pending (after research) |

**Total Research Time:** ~10 hours (Day 1-2 of MVP timeline)

---

## Questions for User (Please Answer)

### PMM Performance:
1. Approximately how long does PMM currently take on Samson? (minutes? hours?)
2. Are there any error messages or warnings when running PMM?
3. Which inference method is currently set: SVI or MCMC?

### SNR Metric:
4. What specifically is "incorrect" about current SNR? (wrong values? crashes? doesn't make sense?)
5. Do you have a reference paper that shows how SNR should be computed for unmixing?
6. For before/after comparison: what should "before" represent? (raw image? initial estimate? something else?)

### Fourier + SVM-RBF:
7. Do you have any reference papers or examples of Fourier-based unmixing?
8. Should this baseline be faster than deep models? (presumably yes, since it's a "low-compute baseline")
9. Is the goal to beat NFINDR baseline, or just provide another classical comparison?

### General:
10. Should I start investigating these issues now, or do you want to provide more context first?
11. Do you have access to the original MiSiCNet / TransNet papers? (might help with SNR definition)

---

**Next Steps:**
Once you answer these questions, I can:
1. Start debugging PMM performance
2. Fix SNR metric with correct definition
3. Design and implement Fourier+SVM-RBF pipeline

Then we proceed with running all experiments and generating results for your presentation.
