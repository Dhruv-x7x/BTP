# MVP Plan - Hyperspectral Unmixing Research
## Target: End of This Week (Presentation Ready)

---

## GOAL
Get all models working correctly, produce complete metrics comparison, basic explainability analysis for a presentation.

---

## 1. CURRENT STATUS

### Models Status
| Model | Samson | Apex | Status | Priority |
|-------|--------|------|--------|----------|
| MiSiCNet | ✅ Tested | ✅ Tested | Working | ✓ |
| TransNet | ✅ Tested | ✅ Tested | Working | ✓ |
| NFINDR | ✅ Tested | ✅ Tested | Working (Baseline) | ✓ |
| PMM (Bayesian) | ⚠️ Tested | ⚠️ Tested | **TOO SLOW - NEEDS FIXING** | **🔥 HIGH PRIORITY** |
| Fourier + SVM-RBF | ❌ Not implemented | ❌ Not implemented | **Needs research & design** | **🔥 HIGH PRIORITY** |

### Critical Issues to Resolve
1. **PMM Performance** - Code likely not correct/optimal, takes too long even with SVI
2. **SNR Metric** - Currently working incorrectly, needs before/after comparison design
3. **Fourier + SVM-RBF Pipeline** - Needs research: how to structure, what Fourier does exactly

---

## 2. MVP SCOPE

### Datasets (2 for MVP)
- ✅ Samson (156 bands, 3 endmembers)
- ✅ Apex (285 bands, 4 endmembers)
- ⏸️ USGS Mineral dataset - **TODO for later**

### Models (5 total)
1. MiSiCNet (CNN) - Working
2. TransNet (Transformer) - Working
3. NFINDR (Geometric Baseline) - Working
4. **PMM (Bayesian) - CRITICAL: Fix performance**
5. **Fourier + SVM-RBF (Classical Baseline) - CRITICAL: Research & implement**

### Metrics (6 core metrics)
1. **RMSE** - Abundance error ✅
2. **SAD** - Spectral Angle Distance for endmembers ✅
3. **SNR** - ⚠️ **BROKEN - needs fixing** (before/after comparison needed, maybe wavelets?)
4. **Entropy** - Spatial entropy of abundances ✅
5. **Time** - Training + inference time ✅
6. **Memory** - Peak memory usage ✅

**NOT needed for MVP:** RE (Reconstruction Error), SID (Spectral Info Divergence), SAM

### Explainability Analysis
**Phase 1 (This week):**
- Understand and document existing visualizations (activations, attention maps)
- Apply similar analysis to all models

**Phase 2 (Later - for paper):**
- Advanced techniques: t-SNE, UMAP, Grad-CAM, layer-wise similarity

### Output Format
- CSV tables with all metrics
- Visualizations in Jupyter notebook
- Summary markdown document

---

## 3. CRITICAL TASKS (Priority Order)

### 🔥 CRITICAL TASK 1: Fix PMM Performance
**Problem:** PMM takes too long even with SVI on T4 GPU
**Hypothesis:** Code is not correct/optimal

**Investigation Steps:**
1. Profile current PMM code to identify bottlenecks
2. Check if using GPU correctly (device placement)
3. Verify SVI parameters (too many steps? inefficient guide?)
4. Compare with PMM paper implementation if available
5. Potential optimizations:
   - Reduce image size / subsample pixels
   - Fewer SVI steps with better convergence monitoring
   - Use mini-batching more efficiently
   - Simplify variational guide (AutoDiagonalNormal instead of AutoLowRankMultivariateNormal?)
   - Check if MCMC is accidentally being used instead of SVI

**Success Criteria:** PMM runs on Samson in <30 minutes on T4

---

### 🔥 CRITICAL TASK 2: Fix SNR Metric
**Problem:** SNR currently working incorrectly

**Questions to Answer:**
1. What is the current implementation doing wrong?
2. What should SNR measure in unmixing context?
   - Option A: SNR of reconstructed image vs original? `SNR = 10*log10(||Y||² / ||Y-Ŷ||²)`
   - Option B: SNR per endmember signal?
   - Option C: SNR of abundances vs ground truth?
3. Before/after comparison - before what? after what?
   - Before unmixing vs after unmixing?
   - Before denoising vs after denoising?
4. Should we use wavelets for SNR calculation?
   - Wavelet decomposition to separate signal from noise at different scales
   - Compute SNR at each wavelet level

**Research Needed:**
- Review hyperspectral unmixing papers to see how SNR is defined
- Check current `compute_snr_per_endmember()` function implementation
- Design proper before/after SNR calculation

**Success Criteria:** Clear SNR definition, correct implementation, meaningful before/after comparison

---

### 🔥 CRITICAL TASK 3: Design & Implement Fourier + SVM-RBF Baseline
**Problem:** Need to research how to structure this pipeline

**Open Questions:**
1. What does Fourier transform do in this context?
   - Extract frequency-domain features from spectra?
   - Denoise spectra before unmixing?
   - Compress spectral information?

2. How do Fourier and SVM-RBF work together?
   - Pipeline A: `Spectrum → FFT → Frequency Features → SVM-RBF → Abundances`
   - Pipeline B: `Spectrum → FFT → Filter → Inverse FFT → SVM-RBF → Abundances`
   - Pipeline C: Something else?

3. How to extract endmembers in this pipeline?
   - Assume endmembers are given (use ground truth)?
   - Use NFINDR to extract endmembers, then Fourier+SVM for abundances?
   - Learn endmembers jointly?

4. SVM-RBF specifics:
   - One SVM per endmember (independent models)?
   - Multi-output SVM?
   - How to enforce sum-to-one constraint?
   - How to enforce non-negativity?

**Research Tasks:**
- Search literature: "Fourier transform hyperspectral unmixing"
- Search literature: "SVM hyperspectral unmixing" or "SVM abundance estimation"
- Look for existing Fourier+SVM pipelines in remote sensing
- Check if scikit-learn SVM supports non-negative + sum-to-one constraints

**Design Decision Needed:** After research, design the exact pipeline architecture

**Success Criteria:** Working Fourier+SVM-RBF baseline that runs on Samson/Apex

---

## 4. EXPLAINABILITY - CONCEPTS & APPROACHES

### What's Already Implemented ✅
From the audit, you already have:

**Activation Visualizations:**
- Extracts intermediate layer outputs from MiSiCNet and TransNet
- Visualizes first 8 channels as heatmaps
- Helps see what features each layer detects
- **Current task:** Understand these visualizations, document what they show

**Attention Maps (TransNet):**
- Shows which image regions transformer attends to
- Per-head attention weights visualized as heatmaps
- Helps understand spatial focus of model
- **Current task:** Understand and document attention patterns

### Advanced Techniques (For Later - Explanation)

#### 1. **Layer-wise Feature Similarity**
**What it measures:** How similar are features between layers?

**How it works:**
- Extract activations from layer 1, layer 2, ..., layer N
- Compute similarity (e.g., CKA, centered kernel alignment) between each pair of layers
- Create similarity matrix showing which layers learn similar features
- High similarity = redundant layers, low similarity = diverse representations

**Use case:** Identify which layers are critical vs redundant, understand feature evolution

**Code example:**
```python
# Extract activations from two layers
act1 = model.layer1_output  # shape: (N, C1, H, W)
act2 = model.layer2_output  # shape: (N, C2, H, W)

# Flatten spatial dimensions
act1_flat = act1.reshape(N, -1)
act2_flat = act2.reshape(N, -1)

# Compute CKA similarity
similarity = centered_kernel_alignment(act1_flat, act2_flat)
# similarity ∈ [0, 1], higher = more similar
```

---

#### 2. **t-SNE / UMAP (Dimensionality Reduction Visualization)**
**What it does:** Projects high-dimensional features to 2D/3D for visualization

**t-SNE (t-distributed Stochastic Neighbor Embedding):**
- Preserves local structure (nearby points in high-D stay nearby in 2D)
- Good for visualizing clusters
- Slower, but very popular

**UMAP (Uniform Manifold Approximation and Projection):**
- Preserves both local and global structure
- Faster than t-SNE
- Better for large datasets

**Use case in unmixing:**
- Visualize how pixels cluster by endmember in feature space
- See if model learns separable representations
- Compare feature spaces across layers

**Code example:**
```python
from sklearn.manifold import TSNE
import umap

# Extract features from a layer (e.g., final layer before output)
features = model.extract_features(X)  # shape: (N_pixels, feature_dim)

# Apply t-SNE
tsne_features = TSNE(n_components=2).fit_transform(features)

# OR apply UMAP (faster)
umap_features = umap.UMAP(n_components=2).fit_transform(features)

# Plot with color = dominant endmember
plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=dominant_endmember_labels)
```

**What to look for:**
- Clear clusters = model learns separable endmember features
- Mixed clusters = model struggles to distinguish endmembers
- Compare early vs late layers to see feature evolution

---

#### 3. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
**What it does:** Highlights which spatial regions contribute most to a specific output

**How it works:**
1. Forward pass: compute output for target endmember
2. Backward pass: compute gradients of endmember output w.r.t. feature maps
3. Weight feature maps by gradient importance
4. Aggregate to produce heatmap showing important regions

**Use case in unmixing:**
- For each endmember, see which image regions are most important
- Verify model focuses on correct spatial areas (e.g., water regions for water endmember)

**Code example:**
```python
# Assume we want to explain endmember k
target_output = model(X)[:, k]  # Abundance of endmember k

# Compute gradients w.r.t. last conv layer
gradients = torch.autograd.grad(target_output.sum(), last_conv_layer)[0]

# Weight feature maps by gradients
weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling
cam = (weights * last_conv_layer).sum(dim=1)  # Weighted sum

# Normalize and overlay on original image
cam = F.relu(cam)  # Only positive contributions
cam = cam / cam.max()  # Normalize to [0, 1]
```

**What to look for:**
- Does Grad-CAM highlight regions where endmember k is truly abundant?
- Mismatches indicate model confusion

---

#### 4. **Saliency Maps**
**What it does:** Shows which input pixels (spectral bands) are most important

**How it works:**
- Compute gradient of output w.r.t. input: `∂output/∂input`
- High gradient = small change in input causes large change in output
- Visualize as heatmap over spectral bands

**Use case in unmixing:**
- Identify which spectral bands are most important for each endmember
- Example: vegetation endmember should highlight red edge + NIR bands

**Code example:**
```python
X.requires_grad = True
output = model(X)[:, k]  # Endmember k

# Compute gradient w.r.t. input
saliency = torch.autograd.grad(output.sum(), X)[0]
saliency = saliency.abs().mean(dim=(0, 2, 3))  # Average over samples & spatial dims

# Plot saliency vs spectral bands
plt.plot(wavelengths, saliency.cpu().numpy())
plt.xlabel('Wavelength (nm)')
plt.ylabel('Importance')
```

**What to look for:**
- Physically meaningful bands highlighted? (e.g., chlorophyll absorption bands for vegetation)

---

### Explainability Roadmap

**This Week (MVP):**
1. Document existing activation visualizations (what do they show?)
2. Document existing attention maps (what patterns emerge?)
3. Apply same visualizations to all models uniformly
4. Create summary notebook with side-by-side comparisons

**Later (Paper):**
- t-SNE/UMAP feature space visualization
- Grad-CAM for spatial importance
- Saliency maps for spectral band importance
- Layer-wise similarity analysis

---

## 5. GOOGLE COLAB OPTIMIZATIONS

### Current Constraints
- **GPU:** T4 with ~15GB VRAM, 12-hour session limit
- **CPU RAM:** 16GB (you mentioned)
- **Disk:** Temporary, cleared after session ends

### Optimization Strategies

#### Strategy 1: Checkpoint After Each Model
**What it means:**
- Save results after each model completes
- If session disconnects, you don't lose everything

**How to implement:**
```python
import pickle

# After MiSiCNet completes
results_misicnet = trainer.run(...)
with open('/content/drive/MyDrive/results_misicnet.pkl', 'wb') as f:
    pickle.dump(results_misicnet, f)

# If session crashes, you can reload:
with open('/content/drive/MyDrive/results_misicnet.pkl', 'rb') as f:
    results_misicnet = pickle.load(f)
```

**Benefit:** Resilience to disconnections

---

#### Strategy 2: Sequential vs Parallel Execution
**Sequential (safer for free Colab):**
```python
# Run one model at a time
results_misicnet = run_misicnet()
results_transnet = run_transnet()
results_nfindr = run_nfindr()
results_pmm = run_pmm()
```

**Pros:** Lower peak memory usage, easier to debug
**Cons:** Longer total time

**We should use sequential** given free T4 constraints.

---

#### Strategy 3: Reduce Memory Footprint
**Techniques:**
1. **Smaller batch size** (if models support batching)
2. **Gradient checkpointing** (trade compute for memory)
3. **Delete intermediate variables:**
   ```python
   results = model.run()
   process_results(results)
   del results  # Free memory immediately
   torch.cuda.empty_cache()
   ```
4. **Run on subsets** (e.g., 50% of Apex image for testing)

---

#### Strategy 4: Monitor Resources
**Add monitoring to notebook:**
```python
import psutil
import torch

def print_memory_stats():
    # CPU RAM
    cpu_mem = psutil.virtual_memory()
    print(f"CPU RAM: {cpu_mem.used/1e9:.2f}GB / {cpu_mem.total/1e9:.2f}GB")

    # GPU RAM
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        gpu_max = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU RAM: {gpu_mem:.2f}GB (peak: {gpu_max:.2f}GB)")

# Call after each model
print_memory_stats()
```

**Benefit:** Identify which models use most memory, optimize accordingly

---

#### Strategy 5: Reduce Number of Runs (If Needed)
**Current:** 5 runs per model per dataset with seeds [101, 202, 303, 404, 505]
**Optimized:** 3 runs per model per dataset (still gives mean ± std)

**Trade-off:** Less statistical robustness, but much faster

**Recommendation:** Start with 3 runs for MVP, can always run more later

---

## 6. THIS WEEK TASK BREAKDOWN

### Day 1-2: Fix Critical Issues
- [ ] **PMM Performance Investigation** (6 hours)
  - Profile code
  - Identify bottlenecks
  - Implement optimizations
  - Test on Samson

- [ ] **SNR Metric Fix** (4 hours)
  - Research correct SNR definition
  - Analyze current implementation
  - Fix and test

- [ ] **Fourier + SVM-RBF Research** (4 hours)
  - Literature review
  - Design pipeline
  - Discuss design decision

### Day 3-4: Implementation & Experiments
- [ ] **Implement Fourier + SVM-RBF** (6 hours)
  - Code pipeline
  - Test on Samson
  - Test on Apex

- [ ] **Run All Models on Both Datasets** (8 hours)
  - MiSiCNet: Samson + Apex (already done, re-run with fixed SNR)
  - TransNet: Samson + Apex (already done, re-run with fixed SNR)
  - NFINDR: Samson + Apex (already done, re-run with fixed SNR)
  - PMM: Samson + Apex (with optimizations)
  - Fourier+SVM: Samson + Apex

- [ ] **Collect All Metrics** (2 hours)
  - RMSE, SAD, SNR, Entropy, Time, Memory
  - Aggregate into CSV tables

### Day 5: Analysis & Presentation Prep
- [ ] **Explainability Analysis** (4 hours)
  - Document existing activation visualizations
  - Document attention maps
  - Apply to all models
  - Create comparison visualizations

- [ ] **Create Comparison Plots** (3 hours)
  - Bar plots: RMSE, SAD, SNR by model
  - Scatter plots: Time vs Accuracy
  - Heatmaps: Metrics across models and datasets
  - Endmember + Abundance visualizations

- [ ] **Summary Document** (2 hours)
  - Markdown with key findings
  - Tables with all metrics
  - Key visualizations
  - Observations and insights

---

## 7. DELIVERABLES FOR PRESENTATION

### 1. Jupyter Notebook (Main)
- All 5 models implemented and running
- Samson + Apex experiments
- Visualizations embedded
- Clean, well-documented code

### 2. Results CSV Files
- `metrics_summary_samson.csv` - All models, all metrics on Samson
- `metrics_summary_apex.csv` - All models, all metrics on Apex
- `metrics_comparison.csv` - Aggregated comparison

### 3. RESULTS.md Document
- Summary tables
- Key findings (which model performs best?)
- Method comparison
- Computational cost analysis

### 4. Visualizations (in notebook)
- Endmember comparison (5 models × 2 datasets)
- Abundance maps (5 models × 2 datasets)
- Metric comparison bar plots
- Activation visualizations
- Attention maps

### 5. Presentation Slides (Optional)
- 10-15 slides summarizing:
  - Problem statement
  - Methods (5 models)
  - Results (metrics comparison)
  - Explainability analysis
  - Conclusions

---

## 8. SUCCESS CRITERIA FOR MVP

✅ **All 5 models working correctly on T4 GPU**
✅ **PMM runs in reasonable time (<30 min on Samson)**
✅ **SNR metric correctly implemented and meaningful**
✅ **Fourier + SVM-RBF baseline implemented**
✅ **Complete metrics tables for Samson + Apex**
✅ **Visualizations comparing all models**
✅ **Basic explainability analysis documented**
✅ **Clean notebook ready for presentation**

---

## 9. WHAT COMES AFTER MVP (For Paper)

### Extensions (Later):
- USGS mineral dataset experiments
- AVIRIS or NEON dataset integration
- Transfer learning experiments (airborne ↔ satellite)
- Advanced explainability (t-SNE, Grad-CAM, etc.)
- Sensitivity analysis
- Wavelet decomposition for SNR
- Statistical significance tests
- Full IEEE paper writing

**Timeline:** Additional 2-4 weeks after MVP for paper-ready results

---

## 10. IMMEDIATE NEXT STEPS (In Order)

1. **Investigate PMM performance issue** (start immediately)
2. **Research SNR metric definition** (parallel task)
3. **Research Fourier + SVM-RBF pipeline design** (parallel task)
4. Discuss findings and design decisions
5. Implement fixes and new baseline
6. Run all experiments
7. Generate metrics and visualizations
8. Prepare presentation materials

---

**Last Updated:** 2025-11-20
**Target Completion:** End of this week
**For:** Presentation (not final paper)
