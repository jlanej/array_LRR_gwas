# Association Engine Design

## Engine Evaluation

The following GWAS engines were evaluated for compatibility with using LRR
as a continuous predictor in genome-wide association.

### PLINK2

Designed for discrete genotype dosages (0/1/2). Can accept dosage
inputs, but continuous LRR values fall outside the expected [0, 2]
range and may trigger warnings or internal clipping. Linear/logistic
models assume genotype coding; residualisation and standard-error
estimation may behave unexpectedly with unbounded continuous
predictors.

**Verdict:** Not directly suitable without non-standard workarounds.

### REGENIE

Two-step method (ridge regression followed by single-variant tests)
that explicitly assumes biallelic genotype coding (0/1/2). Step 1
whole-genome model relies on LD structure and genotype assumptions
that do not apply to LRR. However, the Step 2 single-variant
association scan is essentially WLS/GLS given pre-estimated
variance components — the same principle used by our LMM
implementation.

**Verdict:** Step 1 incompatible; Step 2 concept adopted for LMM.

### TensorQTL

Designed for cis/trans-eQTL mapping (expression ~ genotype). Uses
GPU-accelerated OLS; could in principle be repurposed by swapping the
genotype matrix for LRR. Adds a heavy PyTorch dependency and
requires GPU for performance.

**Verdict:** Technically adaptable but impractical dependency burden.

### Hail

Spark/JVM-based; `linear_regression_rows()` accepts continuous
predictors in principle. Extremely heavy infrastructure requirement
for what reduces to OLS.

**Verdict:** Overkill; not justified for this use case.

### LMM Libraries (limix / glimix-core)

limix and glimix-core implement efficient exact LMMs with REML
estimation. However, they add heavy dependencies (limix pulls in
pandas, dask, xarray). The core algorithm (spectral decomposition
of the GRM + profile REML) is straightforward to implement in
NumPy/SciPy, yielding identical results with no extra dependencies.

**Verdict:** Algorithm adopted; external library not needed.

## Chosen Approach

The primary association method is a **Linear Mixed Model (LMM)** that
accounts for cryptic relatedness via a GRM random effect:

```
phenotype ~ LRR_i + covariates + (1|GRM)
```

The implementation follows the FaST-LMM / EMMA approach:

1. Eigendecompose the GRM once: K = U Λ Uᵀ.
2. Rotate phenotype, covariates, and LRR into the eigen-basis.
3. Estimate the variance-component ratio δ = σ²_e / σ²_g under the
   null model (no marker effect) via profile REML.
4. For each marker, perform weighted least squares in the rotated
   space with weights 1/(λ_i + δ).

This is equivalent to REGENIE Step 2 or fastGWA in concept, but
operates directly on continuous LRR rather than genotype dosages.

OLS and logistic regression remain available as lightweight
alternatives when no GRM is provided.

## Limitations and Best Practices

* The LMM uses the null-model δ for all markers (fast approximation).
  For very strong signals this is slightly conservative.
* Population stratification is controlled via genetic PCs from the
  upstream `compiled_sample_sheet.tsv` (fixed effects).
* Per-marker missing-data handling falls back to a slower loop; ensure
  upstream QC minimises missingness for best performance.
* Logistic regression uses IRLS with a fixed iteration cap; convergence
  failures are silently skipped (NaN beta / p = 1).
