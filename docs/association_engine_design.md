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

## Scalability

The LRR rotation into the GRM eigenbasis (`lrr @ U`) is the most
memory-intensive step.  For a biobank-scale matrix with millions of
markers, materialising the full rotated matrix would cause OOM.

The implementation processes markers in chunks (default: 10 000) so that
only `chunk_size × n_samples` floats are ever resident at once.  The
eigendecomposition and null-model REML fit are performed once; only the
per-marker WLS scan is chunked.

## Logistic Regression and Relatedness

The spectral-decomposition LMM applies only to continuous traits.
Logistic regression (`--method logistic`) uses IRLS without a GRM
random effect.  This means:

* Only fixed-effect covariates (PCs) control for population structure.
* Cryptic relatedness is **not** accounted for.
* Users should pre-filter highly related individuals (e.g. one from
  each pair with kinship > 0.125) before running logistic regression.

Both the API (`run_association`) and the CLI emit a warning when
logistic regression is selected with a GRM present.

## Output Contract for Downstream Segmentation

`AssociationResult.to_records()` returns one dict per variant with a
fixed schema suitable for the downstream HMM/segmentation step
(Issue #5):

| Field        | Type    | Description                              |
|--------------|---------|------------------------------------------|
| `chrom`      | `str`   | Chromosome identifier (e.g. `chr1`)      |
| `pos`        | `int`   | 1-based physical position (bp)           |
| `variant_id` | `str`   | Variant ID or `chrom:pos` fallback       |
| `beta`       | `float` | Effect-size estimate (LRR → phenotype)   |
| `se`         | `float` | Standard error of beta                   |
| `stat`       | `float` | Test statistic (t for OLS/LMM, z for logistic) |
| `p_value`    | `float` | Two-sided p-value                        |
| `n_samples`  | `int`   | Effective sample size for this marker    |
| `method`     | `str`   | `"lmm"`, `"ols"`, or `"logistic"`        |

Records are emitted in the same order as the input `variants` list,
preserving the physical coordinate ordering required by the
segmentation HMM.  The TSV written by the CLI uses tab-delimited
output with headers matching the field names above.
