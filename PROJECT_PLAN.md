# Hyperspectral Unmixing Research Project Plan

## Project Overview
**Goal**: Publish a research paper comparing different hyperspectral unmixing architectures on airborne vs satellite datasets in IEEE format.

**Timeline**: TBD
**Status**: Initial Setup and Verification Phase

---

## 1. CURRENT STATUS (Based on Audit)

### ✅ ALREADY IMPLEMENTED

#### Models (4/4 Core Models)
- [x] **MiSiCNet** (CNN-based) - Implemented with MiSiCTrainer (1,087 lines)
- [x] **TransNet** (Transformer-based) - Implemented with TransNetTrainer
- [x] **NFINDR** (Geometric Baseline) - Implemented with NFINDRRunner (456 lines)
- [x] **PMM** (Bayesian) - Implemented with PMMRunner (582 lines, SVI/MCMC)

#### Datasets (4/7)
- [x] Samson Dataset (156 bands, 3 endmembers: soil, trees, water)
- [x] Apex Dataset (285 bands, 4 endmembers: water, trees, roads, roofs)
- [x] Jasper Ridge Dataset (224 bands)
- [x] Urban Dataset (mentioned in code)
- [ ] EnMAP Dataset (224 bands) - NOT IMPLEMENTED
- [ ] AVIRIS Dataset (224 bands) - NOT IMPLEMENTED
- [ ] NEON Dataset (airborne + satellite) - NOT IMPLEMENTED

#### Metrics (5/6 Core + 2 Advanced)
- [x] RMSE - `numpy_RMSE()`
- [x] SAD - `numpy_SAD()`, `compute_sad()`
- [x] SNR - `compute_snr_per_endmember()`
- [x] Entropy - `compute_entropy()`
- [x] Time/Memory - `report_time_and_memory()`
- [ ] SAM (Spectral Angle Mapper) - **VERIFY if different from SAD**

**Advanced Diagnostics:**
- [x] Activation visualizations (MiSiCNet/TransNet)
- [x] Attention map visualizations (TransNet)

#### Output Formats
- [x] .mat files (MATLAB format)
- [x] .npy files (NumPy arrays)
- [x] CSV result files (per-endmember metrics)
- [x] PNG visualizations (endmembers, abundances, activations)
- [x] LaTeX table generation

---

## 2. RESEARCH OBJECTIVES

### Primary Goal
Compare 4 unmixing methods across airborne vs satellite hyperspectral datasets

### Key Research Questions
1. How do CNN vs Transformer vs Bayesian vs Geometric methods compare?
2. How do models perform differently on airborne vs satellite data?
3. Can models trained on airborne transfer to satellite (and vice versa)?
4. What features are learned at each layer of deep models?
5. Which endmembers are most sensitive to perturbations?

---

## 3. IMPLEMENTATION ROADMAP

### PHASE 1: Verification & Baseline (Week 1-2)
**Status**: Current Phase

#### 1.1 Verify Existing Implementations
- [ ] Run MiSiCNet on Samson dataset (5 runs)
- [ ] Run MiSiCNet on Apex dataset (5 runs)
- [ ] Run TransNet on Samson dataset (5 runs)
- [ ] Run TransNet on Apex dataset (5 runs)
- [ ] Run NFINDR on Samson dataset (5 runs)
- [ ] Run NFINDR on Apex dataset (5 runs)
- [ ] Run PMM on Samson dataset (5 runs)
- [ ] Run PMM on Apex dataset (5 runs)

#### 1.2 Metric Verification
- [ ] Verify all metrics compute correctly
- [ ] Check if SAM is different from SAD (literature review)
- [ ] Ensure before/after SNR is being computed per endmember
- [ ] Test CSV aggregation across multiple runs

#### 1.3 Documentation
- [ ] Update README with project description
- [ ] Document dataset formats and expected structure
- [ ] Create setup instructions for running experiments

**Expected Output**: Baseline results on Samson + Apex for all 4 models

---

### PHASE 2: New Baselines & Methods (Week 3-4)

#### 2.1 Fourier Feature Extraction Baseline
- [ ] Implement FFT-based feature extraction
- [ ] Design simple regression model using Fourier features
- [ ] Test on Samson/Apex
- [ ] Compare computational cost vs deep models

#### 2.2 SVM-RBF Baseline
- [ ] Implement SVM-RBF kernel for abundance regression
- [ ] Feature engineering (spectral indices, derivatives, etc.)
- [ ] Test on Samson/Apex
- [ ] Document limitations vs full unmixing pipeline

#### 2.3 Metric Additions
- [ ] Implement SAM if different from SAD
- [ ] Add RE (Reconstruction Error) tracking if not present
- [ ] Add SID (Spectral Information Divergence) tracking

**Expected Output**: Two additional low-compute baselines with results

---

### PHASE 3: Dataset Expansion (Week 5-6)

#### 3.1 EnMAP Dataset Integration
- [ ] Download/obtain EnMAP dataset
- [ ] Write loader for EnMAP format
- [ ] Verify 224 bands, identify endmembers
- [ ] Add to Data class
- [ ] Test all 4 models on EnMAP

#### 3.2 AVIRIS Dataset Integration
- [ ] Download/obtain AVIRIS dataset
- [ ] Write loader for AVIRIS format
- [ ] Verify 224 bands, identify endmembers
- [ ] Add to Data class
- [ ] Test all 4 models on AVIRIS

#### 3.3 NEON Dataset Integration (Critical for Airborne vs Satellite)
- [ ] Download NEON airborne hyperspectral data
- [ ] Download NEON satellite hyperspectral data (if available)
- [ ] Write loader for NEON format
- [ ] Identify corresponding regions in airborne and satellite data
- [ ] Create paired dataset for transfer learning experiments
- [ ] Add to Data class

**Expected Output**: 3 new datasets integrated, baseline results generated

---

### PHASE 4: Advanced Analysis (Week 7-8)

#### 4.1 Before/After SNR Analysis
- [ ] Modify trainers to save initial SNR (before unmixing)
- [ ] Compute SNR after unmixing per endmember
- [ ] Create comparison tables: SNR_before vs SNR_after
- [ ] Analyze which models improve SNR most
- [ ] Analyze which endmembers benefit most from unmixing

#### 4.2 Sensitivity Analysis on Vegetation
- [ ] Identify vegetation endmembers across datasets
- [ ] Implement perturbation framework (add noise to specific bands)
- [ ] Test model sensitivity to:
  - Red edge band perturbations
  - NIR band perturbations
  - SWIR band perturbations
- [ ] Compute sensitivity metrics (e.g., Jacobian, gradient-based)
- [ ] Generate sensitivity heatmaps per endmember

#### 4.3 Wavelet Decomposition for SNR
- [ ] Implement wavelet decomposition (e.g., Daubechies, Haar)
- [ ] Decompose spectra into approximation + detail coefficients
- [ ] Compute SNR at different wavelet scales
- [ ] Compare multi-scale SNR across models
- [ ] Identify which frequency bands contribute most to unmixing

**Expected Output**: SNR analysis tables, sensitivity heatmaps, wavelet analysis

---

### PHASE 5: Explainability & Feature Analysis (Week 9-10)

#### 5.1 Layer-wise Feature Analysis (Deep Models)
- [ ] Extract activations from all layers of MiSiCNet
- [ ] Extract activations from all layers of TransNet
- [ ] Use dimensionality reduction (t-SNE, UMAP) on activations
- [ ] Visualize feature evolution across layers
- [ ] Correlate layer features with spectral properties

#### 5.2 Attention Analysis (TransNet)
- [ ] Extract attention maps from all transformer blocks
- [ ] Aggregate attention patterns across datasets
- [ ] Identify which spatial regions get most attention
- [ ] Correlate attention with endmember presence
- [ ] Generate attention overlay visualizations

#### 5.3 Explainability Metrics
- [ ] Implement Grad-CAM or similar for CNN (MiSiCNet)
- [ ] Implement attention rollout for Transformer (TransNet)
- [ ] Compute feature importance scores
- [ ] Generate saliency maps for each endmember

#### 5.4 Explainability vs Performance Trade-off
- [ ] Plot accuracy (SAD, RMSE) vs interpretability scores
- [ ] Analyze if simpler models (NFINDR, Fourier) are more interpretable
- [ ] Document which model provides best explanations

**Expected Output**: Feature visualizations, attention maps, explainability analysis

---

### PHASE 6: Transfer Learning Experiments (Week 11-12)

#### 6.1 Airborne → Satellite Transfer
- [ ] Train MiSiCNet on NEON airborne data
- [ ] Fine-tune on NEON satellite data (freeze early layers)
- [ ] Test on NEON satellite data
- [ ] Compare vs training from scratch on satellite
- [ ] Measure performance drop and convergence speed

- [ ] Repeat for TransNet
- [ ] Repeat for PMM (if applicable)

#### 6.2 Satellite → Airborne Transfer
- [ ] Train MiSiCNet on NEON satellite data
- [ ] Fine-tune on NEON airborne data
- [ ] Test on NEON airborne data
- [ ] Compare vs training from scratch on airborne
- [ ] Measure performance drop and convergence speed

- [ ] Repeat for TransNet
- [ ] Repeat for PMM (if applicable)

#### 6.3 Transfer Quantification
- [ ] Define transfer metrics (e.g., % performance retained)
- [ ] Compute feature similarity between source and target layers
- [ ] Identify which layers transfer best
- [ ] Analyze domain shift between airborne and satellite

**Expected Output**: Transfer learning results, domain adaptation analysis

---

### PHASE 7: Statistical Analysis (Week 13)

#### 7.1 Normality Tests
- [ ] Test normality of residuals (Shapiro-Wilk, Anderson-Darling)
- [ ] Test normality of abundance distributions
- [ ] Test normality of per-pixel reconstruction errors
- [ ] Report p-values and Q-Q plots

#### 7.2 Sensitivity Tests
- [ ] Hyperparameter sensitivity (learning rate, batch size, etc.)
- [ ] Input sensitivity (noise injection at various SNR levels)
- [ ] Initialization sensitivity (different random seeds)
- [ ] Band selection sensitivity (drop specific bands)

#### 7.3 Statistical Significance Testing
- [ ] Paired t-tests between model performances
- [ ] ANOVA across all models
- [ ] Bonferroni correction for multiple comparisons
- [ ] Confidence intervals for all metrics

**Expected Output**: Statistical test results, significance tables

---

### PHASE 8: Mineral Data Experiments (Week 14)

#### 8.1 USGS Mineral Data Pipeline
- [ ] Download USGS spectral library (minerals)
- [ ] Create synthetic mineral mixtures
- [ ] Test all 4 models on mineral data
- [ ] Compare performance on minerals vs vegetation

#### 8.2 Cross-Domain Analysis
- [ ] Train on vegetation, test on minerals
- [ ] Train on minerals, test on vegetation
- [ ] Analyze domain gap

**Expected Output**: Mineral unmixing results, cross-domain analysis

---

### PHASE 9: Paper Writing (Week 15-16)

#### 9.1 IEEE Paper Structure
- [ ] Abstract (200-250 words)
- [ ] Introduction (2 pages)
  - Problem statement
  - Motivation (airborne vs satellite comparison)
  - Contributions (4-5 bullet points)
- [ ] Related Work (1.5 pages)
  - Deep learning for unmixing
  - Classical methods
  - Transfer learning in remote sensing
- [ ] Methodology (3 pages)
  - Model descriptions (MiSiCNet, TransNet, PMM, NFINDR)
  - Dataset descriptions
  - Evaluation metrics
  - Experimental setup
- [ ] Results (4 pages)
  - Baseline comparisons (Samson, Apex)
  - Dataset comparisons (airborne vs satellite)
  - Transfer learning results
  - Explainability analysis
  - Statistical tests
- [ ] Discussion (1.5 pages)
  - Key findings
  - Limitations
  - Practical implications
- [ ] Conclusion (0.5 page)
- [ ] References

#### 9.2 Figures and Tables
- [ ] Table 1: Dataset statistics
- [ ] Table 2: Model architectures comparison
- [ ] Table 3: Performance comparison (RMSE, SAD, SNR) on Samson
- [ ] Table 4: Performance comparison on Apex
- [ ] Table 5: Performance comparison on NEON
- [ ] Table 6: Transfer learning results
- [ ] Table 7: Computational cost comparison
- [ ] Figure 1: Example hyperspectral images from datasets
- [ ] Figure 2: MiSiCNet architecture diagram
- [ ] Figure 3: TransNet architecture diagram
- [ ] Figure 4: Endmember comparison (all models, Samson)
- [ ] Figure 5: Abundance maps (all models, Samson)
- [ ] Figure 6: SNR before/after analysis
- [ ] Figure 7: Sensitivity heatmaps
- [ ] Figure 8: Attention visualizations (TransNet)
- [ ] Figure 9: Transfer learning performance curves
- [ ] Figure 10: Explainability comparison

#### 9.3 Paper Refinement
- [ ] First draft completion
- [ ] Internal review and revisions
- [ ] Proofreading
- [ ] LaTeX formatting in IEEE style
- [ ] Final submission-ready version

**Expected Output**: Complete IEEE paper ready for submission

---

## 4. METRICS TO TRACK

### Performance Metrics
| Metric | Description | Implementation Status | Priority |
|--------|-------------|----------------------|----------|
| RMSE | Root Mean Square Error for abundances | ✅ Implemented | HIGH |
| SAD | Spectral Angle Distance for endmembers | ✅ Implemented | HIGH |
| SAM | Spectral Angle Mapper | ⚠️ Verify vs SAD | MEDIUM |
| SNR | Signal-to-Noise Ratio per endmember | ✅ Implemented | HIGH |
| RE | Reconstruction Error | ⚠️ Check | HIGH |
| SID | Spectral Information Divergence | ⚠️ Check | MEDIUM |
| Entropy | Spatial entropy of abundances | ✅ Implemented | MEDIUM |

### Computational Metrics
| Metric | Description | Implementation Status | Priority |
|--------|-------------|----------------------|----------|
| Training Time | Wall-clock time per epoch | ✅ Implemented | HIGH |
| Inference Time | Time per image | ⚠️ Add | HIGH |
| Memory Usage | Peak GPU/CPU memory | ✅ Implemented | MEDIUM |
| FLOPs | Computational complexity | ❌ Not implemented | LOW |

### Explainability Metrics
| Metric | Description | Implementation Status | Priority |
|--------|-------------|----------------------|----------|
| Attention Entropy | Attention distribution sharpness | ❌ Not implemented | MEDIUM |
| Activation Sparsity | Layer activation sparsity | ❌ Not implemented | LOW |
| Grad-CAM Score | Localization accuracy | ❌ Not implemented | MEDIUM |
| Feature Importance | Per-band importance | ❌ Not implemented | MEDIUM |

---

## 5. DATASETS SUMMARY

| Dataset | Bands | Endmembers | Type | Status | Priority |
|---------|-------|------------|------|--------|----------|
| Samson | 156 | 3 (soil, trees, water) | Airborne | ✅ Available | **HIGH** |
| Apex | 285 | 4 (water, trees, roads, roofs) | Airborne | ✅ Available | **HIGH** |
| Jasper Ridge | 224 | ? | Airborne | ✅ Available | MEDIUM |
| Urban | ? | ? | Airborne | ✅ Available | MEDIUM |
| EnMAP | 224 | ? | Satellite | ❌ Need to add | **HIGH** |
| AVIRIS | 224 | ? | Airborne | ❌ Need to add | **HIGH** |
| NEON | ? | ? | Both | ❌ Need to add | **CRITICAL** |

**Note**: NEON dataset is critical for airborne vs satellite comparison.

---

## 6. MODELS SUMMARY

| Model | Type | Architecture | Parameters | Status | Baseline |
|-------|------|--------------|------------|--------|----------|
| MiSiCNet | Deep CNN | Skip-path + conv layers | ~1M (estimate) | ✅ Implemented | No |
| TransNet | Transformer | ViT + AutoEncoder | ~2M (estimate) | ✅ Implemented | No |
| PMM | Bayesian | Dirichlet + SVI/MCMC | N/A | ✅ Implemented | No |
| NFINDR | Geometric | Simplex volume maximization | N/A | ✅ Implemented | **Yes** |
| Fourier+Regressor | Classical | FFT + simple regression | <10K | ❌ To implement | **Yes** |
| SVM-RBF | Classical | RBF kernel + SVM | <100K | ❌ To implement | **Yes** |

---

## 7. EXPERIMENTAL PROTOCOL

### Standard Experimental Setup
- **Runs per model per dataset**: 5 runs with different seeds [101, 202, 303, 404, 505]
- **Metrics computed**: RMSE, SAD, SNR, Entropy, Time, Memory
- **Ordering**: Endmembers and abundances reordered to match ground truth
- **Output**: CSV files with per-run and aggregated (mean ± std) results
- **Visualizations**: Endmembers, abundances, activations (if applicable)

### Validation Strategy
- **Training**: 80% of pixels (random sampling)
- **Validation**: 20% of pixels (random sampling)
- **Testing**: Full image reconstruction

### Hyperparameter Search (if needed)
- Learning rate: [1e-4, 5e-4, 1e-3]
- Batch size: [32, 64, 128]
- Regularization strength: [1e-5, 1e-4, 1e-3]

---

## 8. OPEN QUESTIONS & DECISIONS NEEDED

### Q1: SAM vs SAD
**Question**: Is SAM (Spectral Angle Mapper) different from SAD (Spectral Angle Distance)?
**Action**: Literature review + verify implementation
**Priority**: HIGH

### Q2: NEON Dataset Availability
**Question**: Which NEON sites have paired airborne + satellite data?
**Action**: NEON data portal exploration
**Priority**: CRITICAL

### Q3: Fourier Baseline Design
**Question**: What regression model to use with Fourier features? (Linear, Ridge, Neural Net?)
**Action**: Design decision needed
**Priority**: MEDIUM

### Q4: Transfer Learning Protocol
**Question**: How many layers to freeze? How much fine-tuning data?
**Action**: Experimental design needed
**Priority**: HIGH

### Q5: Explainability Metrics
**Question**: Which explainability metrics are most relevant for remote sensing?
**Action**: Literature review
**Priority**: MEDIUM

### Q6: Mineral Data Source
**Question**: Which USGS mineral library to use? Synthetic or real mixtures?
**Action**: USGS data portal exploration
**Priority**: LOW

---

## 9. RISKS & MITIGATION

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| NEON data not available/accessible | HIGH | LOW | Use alternative paired dataset (e.g., Sentinel + AVIRIS) |
| Models don't transfer well airborne↔satellite | MEDIUM | MEDIUM | Still valuable negative result for paper |
| Computational cost too high for large datasets | MEDIUM | MEDIUM | Use subsampling, smaller image patches |
| Existing implementations have bugs | HIGH | LOW | Thorough verification in Phase 1 |
| Not enough time for all experiments | MEDIUM | HIGH | Prioritize core experiments (Samson, Apex, NEON) |

---

## 10. SUCCESS CRITERIA

### Minimum Viable Paper (MVP)
- ✅ All 4 models working on Samson + Apex
- ✅ Clear performance comparison with statistical tests
- ✅ At least 1 airborne vs satellite comparison (NEON or AVIRIS+EnMAP)
- ✅ Basic explainability analysis (attention maps)
- ✅ 8-10 pages IEEE format paper

### Stretch Goals
- Transfer learning quantification
- Full sensitivity analysis
- Wavelet decomposition analysis
- Mineral data experiments
- 10+ pages IEEE format paper with extensive supplementary material

---

## 11. NEXT IMMEDIATE STEPS

### This Week
1. ✅ Complete project audit (DONE)
2. ✅ Create project plan (DONE)
3. 🔄 Verify MiSiCNet on Samson dataset (5 runs)
4. 🔄 Verify TransNet on Samson dataset (5 runs)
5. 🔄 Verify NFINDR on Samson dataset (5 runs)
6. 🔄 Verify PMM on Samson dataset (5 runs)

### Next Week
7. Run all models on Apex dataset
8. Verify all metrics are correctly computed
9. Research SAM vs SAD difference
10. Begin EnMAP/AVIRIS dataset acquisition

---

## 12. REFERENCES TO GATHER

### Key Papers to Review
- [ ] MiSiCNet original paper
- [ ] TransNet 2022 original paper
- [ ] NFINDR original paper (Winter, 1999)
- [ ] PMM/Bayesian unmixing papers
- [ ] Transfer learning in remote sensing (survey papers)
- [ ] Explainability in hyperspectral imaging
- [ ] Airborne vs satellite comparison studies

---

## Notes
- This is a living document - update as project progresses
- All experiments should be reproducible with documented seeds
- Keep detailed logs of all experimental runs
- Version control all code changes

**Last Updated**: 2025-11-20
**Next Review**: After Phase 1 completion
