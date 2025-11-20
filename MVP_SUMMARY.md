# MVP Summary - Quick Overview

## Goal
Complete a working comparison of 5 unmixing methods by **end of this week** for presentation.

---

## Models (5 Total)

| # | Model | Type | Status | Issue |
|---|-------|------|--------|-------|
| 1 | MiSiCNet | CNN | ✅ Working on Samson/Apex | Need to re-run with fixed SNR |
| 2 | TransNet | Transformer | ✅ Working on Samson/Apex | Need to re-run with fixed SNR |
| 3 | NFINDR | Geometric | ✅ Working on Samson/Apex | Need to re-run with fixed SNR |
| 4 | **PMM** | Bayesian | ⚠️ **TOO SLOW** | **🔥 CRITICAL: Must fix** |
| 5 | **Fourier+SVM** | Classical | ❌ Not implemented | **🔥 CRITICAL: Must implement** |

---

## Datasets (2 for MVP)
- ✅ Samson (156 bands, 3 endmembers)
- ✅ Apex (285 bands, 4 endmembers)
- ⏸️ USGS Mineral (later)

---

## Metrics (6 Core)
1. ✅ RMSE - Abundance error
2. ✅ SAD - Endmember error
3. ⚠️ **SNR - Currently broken, must fix**
4. ✅ Entropy - Spatial entropy
5. ✅ Time - Computation time
6. ✅ Memory - Peak memory usage

**Skip:** RE, SID, SAM

---

## Critical Tasks (Must Complete This Week)

### 🔥 Priority 1: Fix PMM Performance
**Problem:** Too slow even with SVI
**Action:** Profile code, optimize, get running in <30 min on Samson
**Time:** 6 hours

### 🔥 Priority 2: Fix SNR Metric
**Problem:** Currently incorrect
**Action:** Research correct definition, fix implementation, add before/after comparison
**Time:** 4 hours

### 🔥 Priority 3: Implement Fourier + SVM-RBF
**Problem:** Not implemented yet
**Action:** Research design, implement pipeline, test on Samson/Apex
**Time:** 10 hours (4h research + 6h implementation)

### Priority 4: Run All Experiments
**Action:** Run all 5 models on both datasets with all metrics
**Time:** 8 hours

### Priority 5: Explainability Analysis
**Action:** Document existing activations/attention, create comparison visualizations
**Time:** 4 hours

### Priority 6: Results & Presentation
**Action:** Generate CSV tables, plots, summary document
**Time:** 5 hours

**Total:** ~37 hours → Need to work efficiently!

---

## Deliverables

### 1. Working Code
- All 5 models running on Colab T4
- Clean, documented Jupyter notebook

### 2. Results Files
- `metrics_summary_samson.csv`
- `metrics_summary_apex.csv`
- `metrics_comparison.csv`

### 3. Visualizations (in notebook)
- Endmember comparison plots
- Abundance maps
- Metric comparison bar charts
- Activation/attention visualizations

### 4. Documentation
- `RESULTS.md` with summary tables and key findings

---

## Open Questions (Need Answers)

### PMM:
1. How long does it currently take? (minutes/hours?)
2. Which inference method: SVI or MCMC?
3. Any error messages?

### SNR:
4. What exactly is wrong with current SNR?
5. What should "before" and "after" represent?
6. Should we use wavelets?

### Fourier + SVM-RBF:
7. Any reference papers/examples?
8. Should it be faster than deep models?
9. How should pipeline be structured?

### General:
10. Can you share MiSiCNet/TransNet original papers? (for SNR definition)
11. Should I start investigation now?

---

## Timeline (5 Days)

**Day 1-2:** Fix critical issues (PMM, SNR, Fourier+SVM research)
**Day 3-4:** Run experiments, collect metrics
**Day 5:** Analysis, visualizations, presentation prep

---

## Files Created (Ready for Review)

1. ✅ `MVP_PLAN.md` - Detailed plan with explainability concepts explained
2. ✅ `CRITICAL_QUESTIONS.md` - Research questions and investigation plans
3. ✅ `MVP_SUMMARY.md` - This quick reference (you are here)
4. ✅ Updated TODO list - 16 focused tasks

**Please review these files before I commit and push!**

---

## What Happens After MVP?

**Later (for paper):**
- USGS mineral dataset
- AVIRIS or NEON datasets
- Transfer learning experiments
- Advanced explainability (t-SNE, Grad-CAM)
- Statistical tests
- Full IEEE paper writing

**Timeline:** 2-4 additional weeks after MVP

---

## Next Immediate Action

**Waiting for your approval to:**
1. Commit and push these 3 new documents
2. Start investigating PMM performance issue
3. Research SNR definitions
4. Research Fourier+SVM-RBF design

**OR waiting for:**
- Your answers to the questions above
- Any corrections to the plans
- Additional context/clarifications
