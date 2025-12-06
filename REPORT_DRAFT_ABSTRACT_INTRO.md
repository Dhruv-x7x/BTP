# Comparative Analysis of Hyperspectral Unmixing Methods: Deep Learning, Bayesian, and Classical Approaches with Explainability

## Abstract

Hyperspectral imaging captures rich spectral information across hundreds of narrow bands, enabling detailed material identification. However, mixed pixels—containing multiple materials due to limited spatial resolution—pose significant challenges for analysis. Spectral unmixing decomposes these mixed pixels into pure endmember spectra and their corresponding abundance fractions, but existing methods span disparate paradigms with limited cross-comparison and unclear interpretability. We present a comprehensive comparative study of five hyperspectral unmixing methods across three major paradigms: deep learning (MiSiCNet CNN, TransNet Vision Transformer), Bayesian probabilistic (Probabilistic Mixture Model with SVI/MCMC), geometric (NFINDR), and classical machine learning (Fourier transform with SVM-RBF kernel). Our evaluation on standard benchmark datasets (Samson: 156 bands, 3 endmembers; Apex: 285 bands, 4 endmembers) employs six complementary metrics: RMSE, Spectral Angle Distance (SAD), signal-to-noise ratio (SNR), spatial entropy, computational time, and memory usage. We emphasize two critical but underexplored aspects: (1) uncertainty quantification, uniquely provided by Bayesian methods through posterior distributions over endmembers and abundances, and (2) explainability through attention map analysis (TransNet), activation pattern visualization (MiSiCNet), and learned feature characterization. Preliminary results indicate [trade-offs between accuracy and computational cost, with deep learning methods achieving superior accuracy at 10-50× higher computational cost compared to classical baselines]. [TransNet's attention mechanisms reveal spatial focusing patterns corresponding to endmember distributions, while MiSiCNet's convolutional layers learn spectral derivative features.] Our optimized Bayesian implementation achieves 500× speedup (from 267 hours to under 20 minutes) through GPU optimization and mini-batching, making probabilistic methods practical for real-world applications. This work provides evidence-based guidance for method selection and establishes a reproducible framework for future unmixing research with emphasis on interpretability and uncertainty awareness.

**Keywords:** Hyperspectral unmixing, deep learning, Bayesian inference, explainability, Vision Transformer, uncertainty quantification

---

## I. Introduction

### A. Hyperspectral Imaging and Its Applications

Hyperspectral imaging represents a significant advancement in remote sensing technology, capturing spectral information across hundreds of contiguous narrow spectral bands spanning the visible to shortwave infrared spectrum. Unlike traditional RGB imaging with three broad bands or multispectral imaging with 5-10 bands, hyperspectral sensors such as the Airborne Visible/Infrared Imaging Spectrometer (AVIRIS) acquire data in 224 spectral channels with approximately 10 nm spectral resolution [AVIRIS]. This rich spectral signature enables material identification and characterization far beyond what is possible with traditional imaging modalities.

The applications of hyperspectral imaging are diverse and impactful. In mineral exploration, diagnostic absorption features in specific spectral bands enable precise identification of ore deposits and geological formations. Precision agriculture leverages near-infrared and red-edge bands to assess crop health, water stress, and nutrient deficiencies at sub-field scales. Environmental monitoring applications include water quality assessment through detection of algal blooms and pollutants, forest species classification, and land cover mapping. Defense and security applications exploit hyperspectral imaging for target detection and camouflage identification. The fundamental principle underlying all these applications is that different materials exhibit distinct spectral signatures—a unique "fingerprint" determined by their molecular composition and physical structure.

### B. The Mixed Pixel Problem

Despite the spectral richness of hyperspectral data, a fundamental challenge persists: **mixed pixels**. A mixed pixel is one whose measured spectrum arises from multiple distinct materials within the sensor's instantaneous field of view (IFOV). This phenomenon occurs for two primary reasons.

First, **low spatial resolution** is an inherent trade-off in hyperspectral sensor design. To achieve high spectral resolution (narrow bands, high signal-to-noise ratio), sensors must integrate light over longer exposure times or larger spatial areas. Typical hyperspectral sensors have spatial resolutions ranging from 3 to 30 meters per pixel. At these resolutions, a single pixel may encompass multiple land cover types—for example, a pixel containing both a road and adjacent vegetation, or a coastal pixel containing both water and shoreline. The electromagnetic radiation reflected from these diverse materials is integrated by the sensor, yielding a composite spectrum that is a mixture of the constituent materials' individual signatures.

Second, **intimate mixing** occurs when materials are homogeneously mixed at sub-pixel scales. Natural scenes frequently exhibit this phenomenon: soil is composed of mineral grains of varying composition; vegetated areas contain mixtures of leaves, branches, and soil; and urban areas combine concrete, asphalt, metal, and vegetation in complex patterns. When incident solar radiation interacts with such scenes, photons may undergo multiple scattering events among different materials before reaching the sensor. If scattering is primarily linear (single-bounce reflection), linear mixing models apply; if multiple scattering dominates, nonlinear mixing occurs, significantly complicating the unmixing problem.

The consequences of mixed pixels are profound. **Traditional classification approaches** assign each pixel to a single discrete class (e.g., "water," "vegetation," "soil"), implicitly assuming pixel purity. This hard classification paradigm discards valuable sub-pixel information and performs poorly in heterogeneous environments. **Soft classification** methods provide class membership probabilities but do not yield physically interpretable material proportions. These limitations motivated the development of **spectral unmixing**—a quantitative approach to decompose mixed pixel spectra into constituent pure material signatures (endmembers) and their fractional abundances.

### C. Spectral Unmixing: From Classification to Quantitative Analysis

Spectral unmixing addresses the mixed pixel problem by decomposing each observed spectrum into a linear or nonlinear combination of pure endmember spectra, weighted by fractional abundances. In the linear mixing model—the most widely adopted formulation—an observed pixel spectrum is modeled as a weighted sum of endmember spectra plus noise. The abundances represent the proportion of each material within the pixel and are constrained to be non-negative and sum to unity, ensuring physical realizability.

The unmixing problem is typically decomposed into three stages, as formalized by Keshava [Keshava 2002]:

1. **Dimensionality Reduction** (optional): Hyperspectral data cubes are high-dimensional (hundreds of bands), and the curse of dimensionality can adversely affect some algorithms. Dimensionality reduction techniques such as Principal Component Analysis (PCA), Minimum Noise Fraction (MNF), or signal subspace projection methods like HYSIME [Bioucas-Dias & Nascimento 2008] reduce the spectral dimensionality while preserving signal content. However, dimension reduction involves an accuracy-efficiency trade-off; overly aggressive reduction discards informative spectral features.

2. **Endmember Extraction**: The core challenge is to identify the *P* pure endmember spectra present in the scene. This requires estimating the number of endmembers (*virtual dimensionality*) and extracting their spectral signatures. Endmembers may be derived from the data itself (assuming pure pixels exist), from spectral libraries, or learned implicitly by data-driven methods.

3. **Abundance Inversion**: Given the endmember matrix, the final stage solves for the abundance fractions at each pixel. This inverse problem is typically ill-posed and requires regularization or constraints (non-negativity, sum-to-one) to obtain meaningful solutions.

### D. Existing Approaches: A Multi-Paradigm Landscape

Spectral unmixing methods span multiple paradigms, each with distinct assumptions, strengths, and limitations.

#### 1) Geometric Methods

Geometric methods exploit a fundamental insight: if all pixels in a hyperspectral image are linear mixtures of *P* endmembers, then in spectral space, the data points lie within a *P*-dimensional simplex whose vertices correspond to the endmembers. Geometric algorithms seek to identify these vertices.

**Pixel Purity Index (PPI)** projects the data onto random directions and identifies pixels that are extremal in these projections, under the assumption that pure pixels lie at the data cloud's boundary. **N-FINDR** [Winter 1999] formulates endmember extraction as a simplex volume maximization problem: the correct set of endmembers defines the simplex of maximum volume enclosing all data points. The algorithm iteratively replaces candidate endmembers to increase simplex volume, converging to a local maximum. **Vertex Component Analysis (VCA)** [Nascimento & Dias 2005] projects the data orthogonally onto the subspace orthogonal to previously identified endmembers, sequentially extracting vertices.

**Advantages**: Geometric methods are fast, deterministic (aside from random initialization), require no training data, and can identify low-probability endmembers (rare materials present in few pixels). **Disadvantages**: They are sensitive to outliers and noise, assume that pure pixels exist in the data (the "pure pixel assumption"), and provide no mechanism for uncertainty quantification.

#### 2) Statistical and Probabilistic Methods

Statistical methods model spectral unmixing in a probabilistic framework, treating endmembers and abundances as random variables with associated probability distributions.

**Gaussian Mixture Models (GMM)** assume that observed spectra arise from a mixture of Gaussian distributions, with mixing coefficients corresponding to abundances. Parameter estimation is typically performed via Expectation-Maximization (EM). However, GMMs do not naturally enforce the sum-to-one constraint on abundances and may not align well with the physical mixing process.

**Bayesian methods**, particularly the **Probabilistic Mixture Model (PMM)** [Hoidn et al.], adopt a fully Bayesian approach with prior distributions over endmembers (e.g., Gaussian priors) and abundances (e.g., Dirichlet priors, which naturally enforce non-negativity and sum-to-one constraints). The key advantage is that Bayesian inference yields **full posterior distributions** over all unknowns, explicitly quantifying both **epistemic uncertainty** (uncertainty due to lack of knowledge, such as the number of endmembers) and **aleatoric uncertainty** (inherent randomness such as sensor noise). Inference is performed via Markov Chain Monte Carlo (MCMC) methods, specifically Hamiltonian Monte Carlo with the No-U-Turn Sampler (HMC-NUTS) [Hoffman & Gelman 2014], or via faster approximate methods such as Stochastic Variational Inference (SVI) [Kucukelbir et al. 2017].

**Advantages**: Bayesian methods provide uncertainty quantification (confidence intervals on endmembers and abundances), are robust to noise, and do not require pure pixels. **Disadvantages**: MCMC inference is computationally expensive (our initial implementation required 267 hours for a single dataset), and results are sensitive to prior specification. Recent work has focused on optimizing Bayesian unmixing for practical use.

#### 3) Deep Learning Methods

The availability of large hyperspectral datasets and advances in deep learning have motivated neural network-based unmixing methods, which learn endmembers and abundances end-to-end from data.

**Convolutional Neural Networks (CNNs)**: **MiSiCNet** [Rasti et al. 2020] employs a dual-path architecture with skip connections. The main path consists of convolutional layers with batch normalization and LeakyReLU activations, while a skip path preserves fine-scale features. The network jointly learns an endmember matrix and per-pixel abundance maps through a composite loss function that balances reconstruction error, sparsity regularization (via orthogonal subspace projection or nuclear norm), and sum-to-one constraints. Autoencoder-based approaches similarly encode input spectra into a latent abundance space and decode back to spectral space.

**Vision Transformers**: **TransNet** [Palsson et al. 2022] adapts the Vision Transformer (ViT) architecture to hyperspectral unmixing. Hyperspectral images are divided into spatial patches, which are linearly embedded and augmented with positional encodings. Multi-head self-attention layers capture long-range spatial and spectral dependencies, and cross-attention mechanisms relate pixel spectra to learned endmember representations. An autoencoder decoder maps the transformer output to abundance maps, with softmax normalization ensuring sum-to-one constraints.

**Advantages**: Deep learning methods learn complex, nonlinear patterns directly from data without hand-crafted features, achieve state-of-the-art accuracy on benchmark datasets, and scale well to large datasets. **Disadvantages**: They require substantial training data, are computationally expensive (training takes hours on GPUs), lack interpretability (models are "black boxes"), and provide no inherent uncertainty estimates (though recent work on Bayesian neural networks addresses this).

#### 4) Classical Machine Learning Baselines

Kernel-based methods and frequency-domain techniques represent a middle ground between geometric methods and deep learning.

**Support Vector Machines (SVM)** with Radial Basis Function (RBF) kernels have been applied to hyperspectral classification and, more recently, to abundance regression [Plaza et al.]. SVMs map input spectra to high-dimensional feature spaces where linear separation is possible. For unmixing, one SVM regressor per endmember predicts abundances, with post-processing to enforce physical constraints.

**Fourier Transform methods** leverage the observation that spectral signatures contain information in the frequency domain. Fast Fourier Transform (FFT) provides dimensionality reduction by retaining low-frequency components (which capture broad spectral shape) and discarding high-frequency noise. FFT-based unmixing has been shown to be faster than PCA-based approaches while maintaining high accuracy [Mount et al.].

**Hybrid approaches** combine geometric endmember extraction (e.g., NFINDR) with machine learning-based abundance estimation (e.g., SVM-RBF), offering a balance between interpretability and performance.

**Advantages**: Classical ML methods are fast (training in minutes, inference in seconds), work with limited training data, and are relatively interpretable. **Disadvantages**: They have a performance ceiling compared to deep learning and may require careful feature engineering.

### E. Knowledge Gaps and Motivation

Despite extensive research, several critical gaps remain in the spectral unmixing literature:

**Gap 1: Limited Cross-Paradigm Comparisons**. Most studies compare methods within a single paradigm (e.g., CNN variant A vs. CNN variant B, or NFINDR vs. VCA). Comprehensive evaluations spanning geometric, statistical, classical ML, and deep learning methods are rare. Consequently, it is unclear which paradigm performs best under specific conditions (data size, noise level, number of endmembers, computational budget).

**Gap 2: Unclear Computational Trade-offs**. Deep learning methods are reputed to be accurate but slow. Bayesian methods are known to quantify uncertainty but are computationally prohibitive. Classical methods are fast but assumed to be less accurate. However, quantitative comparisons of accuracy vs. computational cost (training time, inference time, memory usage) across paradigms are scarce. Practitioners lack evidence-based guidance on which method to deploy given their constraints.

**Gap 3: Explainability and Interpretability Deficit**. Deep learning methods—particularly CNNs and Transformers—are black boxes. We do not understand *what* spectral or spatial features these models learn, *why* they make specific predictions, or *how* they differ from interpretable geometric or statistical methods. For Vision Transformers, attention mechanisms offer a window into model reasoning, but systematic analysis of attention patterns in the context of unmixing is absent. Similarly, convolutional filter activations have not been thoroughly examined to understand what spectral features (e.g., absorption bands, derivatives) the network detects.

**Gap 4: Underexplored Recent Architectures**. TransNet, proposed in 2022, applies Vision Transformers to unmixing but has not been widely benchmarked against other methods. The potential advantages of global self-attention over local convolution for capturing spectral correlations remain underexplored.

**Gap 5: Lack of Uncertainty Quantification**. In real-world applications, knowing not only the estimated abundances but also the confidence in those estimates is critical. Bayesian methods naturally provide this through posterior distributions, but adoption has been limited due to computational cost. Optimizing Bayesian unmixing for practical use is an open challenge.

**Why This Study Matters**: This work addresses these gaps by providing a unified, reproducible evaluation of five representative methods across three paradigms, with emphasis on two underexplored dimensions—**uncertainty quantification** and **explainability**. Our results inform practitioners on method selection based on accuracy, speed, and interpretability requirements, and provide researchers with a standardized framework and baseline for future work.

### F. Research Questions

This study investigates the following research questions:

**RQ1 (Accuracy)**: How do deep learning methods (MiSiCNet CNN, TransNet Vision Transformer) compare to classical methods (NFINDR geometric, PMM Bayesian, Fourier+SVM) in terms of endmember accuracy (Spectral Angle Distance) and abundance accuracy (RMSE)?

**RQ2 (Computational Cost vs. Accuracy Trade-offs)**: What are the computational costs (training time, inference time, memory usage) of each method, and how do these scale with dataset size and spectral dimensionality? Can we quantify the accuracy-efficiency Pareto frontier?

**RQ3 (Transformer vs. CNN)**: Does TransNet's Vision Transformer architecture provide measurable advantages over MiSiCNet's CNN architecture? Do global self-attention mechanisms capture long-range spectral dependencies more effectively than local convolutions?

**RQ4 (Classical Baseline Competitiveness)**: Can a simple classical baseline—Fourier transform for feature extraction combined with SVM-RBF for abundance regression—provide competitive performance at significantly lower computational cost?

**RQ5 (Explainability and Interpretability)**: What can we learn about how these models work?
- For TransNet: What spatial or spectral regions do attention heads focus on? Do attention patterns correspond to endmember distributions?
- For MiSiCNet: What spectral features (e.g., absorption bands, derivatives) do convolutional filters learn across layers?
- Comparative: How do learned features differ from hand-crafted features (e.g., NFINDR's simplex volume, Fourier frequencies)?

**RQ6 (Uncertainty Quantification)**: How much uncertainty exists in endmember and abundance estimates? Does the Bayesian PMM's posterior distribution width correlate with estimation errors? Can uncertainty-aware predictions improve downstream decision-making?

*(Optional for future work)*: **RQ7 (Generalization)**: How does performance vary with number of endmembers, spectral dimensionality, and noise levels? Do deep learning methods overfit to specific datasets?

### G. Contributions

This work makes the following contributions:

**1. Comprehensive Cross-Paradigm Comparison**: We provide the first unified evaluation of five hyperspectral unmixing methods spanning geometric (NFINDR), Bayesian probabilistic (PMM with SVI/MCMC), CNN-based deep learning (MiSiCNet), Transformer-based deep learning (TransNet), and classical machine learning (Fourier+SVM-RBF) paradigms. All methods are evaluated on identical datasets (Samson, Apex) using a standardized protocol (same train/test splits, same random seeds, same metrics).

**2. Emphasis on Uncertainty Quantification and Explainability**: Unlike prior work focused solely on accuracy metrics, we emphasize:
   - **Uncertainty quantification** through Bayesian PMM's posterior distributions over endmembers and abundances, providing confidence intervals and prediction credibility assessments.
   - **Explainability analysis** through systematic examination of TransNet's attention maps (revealing spatial focus patterns), MiSiCNet's convolutional activations (revealing learned spectral features), and comparison with interpretable classical methods.

**3. Optimized Bayesian Inference**: We identify and resolve critical performance bottlenecks in PMM implementation, achieving a **500× speedup** (from 267 hours to under 20 minutes per dataset) through GPU optimization, mini-batching (subsampling pixels per SVI step), simplified variational guides, and early stopping. This makes Bayesian methods practical for real-world hyperspectral analysis.

**4. Novel Classical Baseline**: We propose and evaluate a hybrid classical pipeline: NFINDR for endmember extraction, followed by FFT-based feature extraction (dimensionality reduction from *L* bands to *K* frequency components) and SVM-RBF regression for abundance estimation. This pipeline provides a fast, interpretable baseline that does not require large training datasets.

**5. Standardized Evaluation Protocol**: We establish a reproducible evaluation framework:
   - **6 complementary metrics**: RMSE (abundance error), SAD (endmember error), SNR (signal-to-noise ratio before/after unmixing), spatial entropy (abundance map sharpness), training/inference time, memory usage.
   - **Multi-run experiments**: 5 independent runs with different random seeds, reporting mean and standard deviation.
   - **Open-source implementation**: All code, trained models, and results publicly available [to be released].

**6. Practical Guidance**: Based on our results, we provide a method selection flowchart and trade-off analysis to guide practitioners in choosing an appropriate unmixing method given their accuracy requirements, computational budget, and interpretability needs.

### H. Report Organization

The remainder of this report is organized as follows:

- **Section II (Methodology)** describes the five unmixing methods in detail, including model architectures, loss functions, and hyperparameters. We also describe the datasets, evaluation metrics, and experimental protocol.

- **Section III (Experiments and Implementation)** details the experimental setup, including training procedures, hardware configuration, and software dependencies. We also present implementation challenges encountered (e.g., PMM performance bottlenecks) and their solutions.

- **Section IV (Results)** presents quantitative performance comparisons across all methods and metrics, including statistical significance testing. We also present qualitative results: attention map visualizations for TransNet, activation patterns for MiSiCNet, and uncertainty quantification for PMM.

- **Section V (Explainability Analysis)** provides in-depth analysis of what each model learns and how it makes predictions, with emphasis on attention mechanisms, learned features, and uncertainty estimates.

- **Section VI (Discussion)** interprets our findings, discusses limitations, and contextualizes results within the broader unmixing literature. We also discuss practical implications for method selection.

- **Section VII (Future Work and Conclusion)** outlines directions for future research, including extensions to additional datasets (NEON airborne/satellite, USGS minerals), transfer learning experiments, and advanced explainability techniques. We conclude with a summary of key findings and their implications.

---

**[Potential Figure Placements]**

- **Figure 1** (after Section I.B): Illustration of the mixed pixel problem.
  - Left: High-resolution RGB image showing road + vegetation boundary
  - Right: Low-resolution hyperspectral pixel covering both materials, with composite spectrum

- **Figure 2** (after Section I.C): Schematic of the spectral unmixing problem.
  - Input: Hyperspectral cube (rows × cols × bands)
  - Output: Endmember matrix (P × L) + Abundance maps (rows × cols × P)
  - Equation: Y = A · E + W (in box)

- **Figure 3** (after Section I.D.1): Geometric interpretation of NFINDR.
  - 3D simplex (tetrahedron) enclosing data points in PCA-reduced space
  - Vertices labeled as endmembers

- **Figure 4** (after Section I.D.3): Architecture diagrams.
  - Top: MiSiCNet CNN with skip path and main path
  - Bottom: TransNet Vision Transformer with patch embedding and attention blocks

---

