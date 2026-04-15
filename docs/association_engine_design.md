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
* Missing LRR values are mean-imputed per marker so that the full-sample
  GRM eigenbasis can be reused for all markers.  This avoids an O(N³)
  eigendecomposition per marker and is the standard approach in EMMA,
  SAIGE, and fastGWA.  Ensure upstream QC minimises missingness for best
  accuracy.
* Logistic regression uses IRLS with a fixed iteration cap; convergence
  failures are silently skipped (NaN beta / p = 1).  The per-marker
  Python loop is a known bottleneck for very large marker counts.

## GRM Marker Selection and LD Pruning

The GRM aims to model neutral background relatedness between samples.
For the `plink2` backend (default), the full pipeline proceeds without
ever loading a dosage matrix into Python:

1. **Lightweight metadata scan** — `read_variant_metadata()` reads variant
   IDs and sample names from the BCF header + records without decoding
   `FORMAT/GT` dosage values.
2. **Upstream variant QC** — Variant IDs from `collated_variant_qc.tsv`
   (provided via `--variant-qc`) are used to build the QC-passing ID list.
   Variants that fail call rate, HWE, or MAF thresholds across all
   ancestries are excluded.  When no QC file is provided, plink2's own
   `--maf` and `--geno` filters are applied.
3. **BCF → plink2 BED** — `make_plink2_bed()` converts the BCF to a
   plink1-format BED/BIM/FAM fileset, filtering to only QC-passing
   variants and analyzed samples in a single plink2 invocation.  The BED
   is stored in `--audit-dir` for provenance (or a temp dir if omitted).
4. **LD pruning on BED** — `ld_prune_plink2()` runs
   `plink2 --indep-pairwise` on the BED (not the BCF), which is faster
   because the binary format is already decoded.
5. **GRM from LD-pruned BED** — `compute_grm_plink2()` runs
   `plink2 --make-grm-bin` on the BED with an extract list of
   LD-pruned IDs, producing the GCTA-format GRM without materialising
   any dosage array in Python.

This pipeline ensures that:
- The full genotype dosage matrix never enters Python memory.
- Each variant passes both upstream ancestry-stratified QC *and* plink2
  MAF/call-rate checks (no double-filtering when QC IDs are provided).
- Provenance is captured: BED/BIM/FAM files in `--audit-dir` record
  exactly which markers and samples entered the GRM.

Two backends are available:

| Backend  | When to use                                              |
|----------|----------------------------------------------------------|
| `plink2` | **Default.** Fully plink2-native; no Python dosage       |
|          | matrix.  Requires `plink2` on `$PATH`.  Exits with an   |
|          | error if plink2 is not found.                            |
| `numpy`  | Explicit fallback via `--ld-backend numpy`.  Loads the   |
|          | full dosage matrix into Python.  No external tools.      |

CLI flags:

| Flag              | Default     | Description                          |
|-------------------|-------------|--------------------------------------|
| `--no-ld-prune`   | off         | Disable LD pruning entirely          |
| `--ld-window-bp`  | 1 000 000   | Window size in bp                    |
| `--ld-r2-thresh`  | 0.2         | r² threshold                         |
| `--ld-backend`    | `plink2`    | Backend (`numpy` or `plink2`)        |

## Scalability

The association pipeline uses a **streaming architecture** that never holds
the full sample × marker LRR matrix in RAM:

1. **Metadata scan** — `read_variant_metadata()` reads only variant-level
   metadata (chrom, pos, id, intensity_only flag) and sample IDs from the
   BCF header.  No `FORMAT/LRR` values are loaded.
2. **Metadata-based pre-filter** — INTENSITY_ONLY markers are excluded via
   a boolean mask *before* any LRR values are read.
3. **Chunked LRR streaming** — `stream_lrr_chunks()` yields
   `(lrr_chunk, variants_chunk)` tuples with at most `chunk_size` markers
   each.  `sample_mask` and `variant_mask` ensure that only the relevant
   subset of the BCF is decoded.
4. **Streaming association** — `run_association_streaming()` consumes the
   chunk generator, applies per-chunk monomorphic-LRR filtering inline,
   and accumulates per-marker results without materialising the full
   matrix.

Peak RAM is bounded by `chunk_size × n_samples` floats (plus the GRM
when using LMM).  For the default chunk size of 5 000 and 10 000 samples,
this is about 400 MB — well within typical workstation limits.

## Sex-Chromosome Analysis

The `--sex-chr-mode` CLI option runs additional association scans on sex
chromosomes.  Four modes are supported:

| Mode | Chromosome | Sample subset | Extra covariates |
|------|-----------|--------------|-----------------|
| `x_with_sex_covariate` | chrX | All | Sex (binary) |
| `x_male_only` | chrX | Males only | — |
| `x_female_only` | chrX | Females only | — |
| `y_male_only` | chrY | Males only | — |

Each mode writes a separate TSV file alongside the main output
(e.g. `results.x_male_only.tsv`).  Sex-chromosome scans always use OLS
(no GRM) because the autosomal GRM is not appropriate for sex-linked
markers.

Requires `--sample-sheet` with a `predicted_sex` column (1 = male,
2 = female).  Modes with fewer than 3 qualifying samples are skipped
with a warning.

## Logistic Regression and Relatedness

The spectral-decomposition LMM applies only to continuous traits.
Logistic regression (`--method logistic`) uses IRLS without a GRM
random effect.  This means:

* Only fixed-effect covariates (PCs) control for population structure.
* Cryptic relatedness is **not** accounted for.
* Users should pre-filter highly related individuals (e.g. one from
  each pair with kinship > 0.125) before running logistic regression,
  or consider `--method lmm` as a common continuous-trait approximation
  for binary phenotypes.

Both the API (`run_association`, when a GRM is provided) and the CLI
emit warnings that logistic regression does not apply the GRM random
effect.

When users intentionally run `--method lmm` with a binary phenotype
(0/1) as a continuous-trait approximation, the engine applies a safety
heuristic based on case fraction. If `#cases / #valid_samples` falls
outside `[0.10, 0.90]`, it emits a strong warning that continuous LMM
on a highly unbalanced binary trait can inflate Type I error,
particularly for lower-frequency variants. In that setting, users
should either pre-filter related individuals and use logistic
regression, or focus interpretation on variants satisfying
`(MAF × #cases) > 100`.

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

When `--variant-qc` is provided, three additional boolean columns are
appended for provenance tracking:

| Field | Type | Description |
|---|---|---|
| `all_ancestries_call_rate_pass` | `bool` | Cross-ancestry call-rate QC flag |
| `all_ancestries_hwe_pass` | `bool` | Cross-ancestry HWE QC flag |
| `all_ancestries_maf_pass` | `bool` | Cross-ancestry MAF QC flag |
