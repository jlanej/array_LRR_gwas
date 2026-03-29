# array\_LRR\_gwas

**Genome-wide association analysis using Log R Ratio as a quantitative copy-number proxy**

[![CI](https://github.com/jlanej/array_LRR_gwas/actions/workflows/ci.yml/badge.svg)](https://github.com/jlanej/array_LRR_gwas/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Table of Contents

- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [Pipeline Summary](#pipeline-summary)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running on HPC with Apptainer](#running-on-hpc-with-apptainer)
- [CLI Reference](#cli-reference)
  - [correct](#correct)
  - [associate](#associate)
  - [segment](#segment)
- [QC Configuration](#qc-configuration)
- [Input Formats](#input-formats)
- [Output Formats](#output-formats)
- [Python API](#python-api)
- [Methodology](#methodology)
  - [Batch-Effect Correction](#batch-effect-correction)
  - [Association Analysis](#association-analysis)
  - [Segmentation](#segmentation)
- [Limitations and Caveats](#limitations-and-caveats)
- [Upstream Dependencies](#upstream-dependencies)
- [Development](#development)
- [References](#references)
- [License](#license)

---

## Overview

`array_lrr_gwas` is a Python toolkit for performing **GWAS using array-based
Log R Ratio (LRR) values as continuous copy-number proxies**.  Standard GWAS
engines (PLINK2, REGENIE, Hail) are designed for discrete genotype dosages
(0/1/2) and do not natively support unbounded continuous predictors like LRR.
This package fills that gap by providing:

1. **Batch-effect correction** — removes array-scanning artefacts from the LRR
   matrix via randomised SVD, so that downstream association reflects true
   biology rather than technical noise.
2. **Association analysis** — tests each marker's corrected LRR against a
   phenotype using a Linear Mixed Model (LMM) that accounts for cryptic
   relatedness, with OLS and logistic alternatives.
3. **Segmentation** — collapses per-marker association statistics into
   CNV-associated genomic intervals using a two-state Hidden Markov Model or
   threshold-and-merge strategy.

The entire pipeline runs from the command line or as a Python library, is
containerised via Docker / Apptainer for reproducible HPC deployment, and has
no dependencies beyond NumPy, SciPy, scikit-learn, pysam, and PyYAML.

---

## Scientific Background

Genotyping arrays (e.g. Illumina Infinium) report a **Log R Ratio (LRR)** for
every probe — the log₂-scaled normalised total signal intensity relative to
the expected diploid cluster position.  Deviations from zero indicate
copy-number changes: deletions drive LRR negative, duplications drive it
positive.

Treating LRR as a **quantitative, continuous predictor** in a GWAS framework
enables association testing of copy-number variation (CNV) effects on phenotype
at probe-level resolution, without requiring discrete CNV calling.  This
approach is more sensitive to small or mosaic dosage shifts than hard
gain/loss calls.

However, LRR values are heavily affected by **batch effects** — systematic
artefacts from array scanning date, scanner, reagent lot, and plate position.
These artefacts are approximately low-rank and must be removed before
association testing.  This package addresses that need.

---

## Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│  Upstream: jlanej/illumina_idat_processing                         │
│  IDAT → BCF (FORMAT/GT, FORMAT/LRR, FORMAT/BAF)                   │
│  + compiled_sample_sheet.tsv (QC metrics, ancestry PCs)            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 1: Batch-Effect Correction  (array-lrr-gwas correct)          │
│                                                                      │
│  ① Detect genome build (GRCh37 / GRCh38 / T2T-CHM13)               │
│  ② Classify samples → HQ (LRR_SD ≤ 0.35, call_rate ≥ 0.97) / LQ   │
│  ③ Subset markers (call-rate, variance, complexity filters)         │
│  ④ Truncated SVD on HQ sub-matrix → batch PCs                      │
│  ⑤ Select k components (Marchenko-Pastur or manual)                │
│  ⑥ Project LQ samples onto HQ loadings                             │
│  ⑦ Residualise LRR matrix → corrected BCF                          │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 2: Association Analysis  (array-lrr-gwas associate)           │
│                                                                      │
│  ① Load corrected LRR + phenotype + covariates (PCs)               │
│  ② Compute GRM from genotypes (for LMM)                            │
│  ③ LMM: spectral decomposition → REML δ → per-marker WLS          │
│     OR OLS / logistic regression (no GRM)                           │
│  ④ Write per-marker results TSV                                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 3: Segmentation  (array-lrr-gwas segment)                     │
│                                                                      │
│  ① Read per-marker association TSV                                  │
│  ② Two-state HMM on -log₁₀(p) per chromosome  (or threshold-merge)│
│  ③ Collect runs of associated state → genomic intervals            │
│  ④ Write BED with interval statistics                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### From Source (pip)

```bash
git clone https://github.com/jlanej/array_LRR_gwas.git
cd array_LRR_gwas
pip install -e '.[dev]'      # editable install with test dependencies
```

### Docker

```bash
docker pull ghcr.io/jlanej/array_lrr_gwas:latest
docker run --rm ghcr.io/jlanej/array_lrr_gwas:latest --help
```

Or build locally:

```bash
docker build -t array-lrr-gwas .
```

### Apptainer / Singularity (HPC)

```bash
apptainer pull array_lrr_gwas.sif docker://ghcr.io/jlanej/array_lrr_gwas:latest
```

### Requirements

- Python ≥ 3.10
- NumPy ≥ 1.22
- SciPy ≥ 1.8
- scikit-learn ≥ 1.1
- pysam ≥ 0.20
- PyYAML ≥ 6.0
- **Optional:** [fbpca](https://github.com/facebookarchive/fbpca) for an
  alternative SVD backend

---

## Quick Start

```bash
# Step 1 — Remove batch effects
array-lrr-gwas correct input.bcf -o corrected.bcf --build GRCh38 -v

# Step 2 — Run LMM association
array-lrr-gwas associate corrected.bcf \
    --phenotype pheno.tsv \
    --sample-sheet compiled_sample_sheet.tsv \
    --genotype-bcf genotypes.bcf \
    --method lmm \
    -o results.tsv -v

# Step 3 — Segment into CNV regions
array-lrr-gwas segment results.tsv -o regions.bed -v
```

**Best-practice workflow with upstream QC filtering:**

```bash
# Step 1 — Remove batch effects with upstream variant QC
array-lrr-gwas correct input.bcf -o corrected.bcf --build GRCh38 \
    --variant-qc collated_variant_qc.tsv -v

# Step 2 — Run LMM association with QC-filtered GRM
array-lrr-gwas associate corrected.bcf \
    --phenotype pheno.tsv \
    --sample-sheet compiled_sample_sheet.tsv \
    --genotype-bcf genotypes.bcf \
    --variant-qc collated_variant_qc.tsv \
    --method lmm \
    -o results.tsv -v

# Step 3 — Segment into CNV regions
array-lrr-gwas segment results.tsv -o regions.bed -v
```

---

## Running on HPC with Apptainer

Apptainer (formerly Singularity) is the recommended way to run
`array_lrr_gwas` on shared HPC clusters.

### Pull the Container Image

```bash
# Pull once to a shared location (e.g. a project directory)
apptainer pull /path/to/containers/array_lrr_gwas.sif \
    docker://ghcr.io/jlanej/array_lrr_gwas:latest
```

### Full Pipeline Example (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=lrr_gwas
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=lrr_gwas_%j.log

SIF=/path/to/containers/array_lrr_gwas.sif

INPUT_BCF=/data/project/genotypes.bcf
VARIANT_QC=/data/project/collated_variant_qc.tsv
PHENO=/data/project/phenotype.tsv
SAMPLE_SHEET=/data/project/compiled_sample_sheet.tsv
GT_BCF=/data/project/genotypes.bcf
OUTDIR=/data/project/results
mkdir -p "${OUTDIR}"

# Step 1: Batch-effect correction
apptainer run --bind /data "${SIF}" correct \
    "${INPUT_BCF}" \
    -o "${OUTDIR}/corrected.bcf" \
    --build GRCh38 \
    --variant-qc "${VARIANT_QC}" \
    -v

# Step 2: Association (LMM with GRM and 20 ancestry PCs)
# LD pruning is enabled by default for GRM computation.
apptainer run --bind /data "${SIF}" associate \
    "${OUTDIR}/corrected.bcf" \
    --phenotype "${PHENO}" \
    --sample-sheet "${SAMPLE_SHEET}" \
    --genotype-bcf "${GT_BCF}" \
    --variant-qc "${VARIANT_QC}" \
    --method lmm \
    --n-pcs 20 \
    -o "${OUTDIR}/association_results.tsv" \
    -v

# Step 3: Segmentation (HMM, default parameters)
apptainer run --bind /data "${SIF}" segment \
    "${OUTDIR}/association_results.tsv" \
    -o "${OUTDIR}/cnv_regions.bed" \
    -v
```

> **Note:** Use `--bind` to make host file systems visible inside the
> container. Adjust paths to match your cluster's storage layout. For
> multiple directories, use comma-separated paths (for example,
> `--bind /data,/scratch`).

### Per-Chromosome Parallelisation

For large biobank-scale datasets, the association step can be
parallelised by subsetting the input BCF per chromosome and running
separate jobs.  The segmentation step already operates per-chromosome
internally.

---

## CLI Reference

### `correct`

Remove batch effects from array LRR values via truncated SVD.

```
array-lrr-gwas correct INPUT -o OUTPUT [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `INPUT` | path | *required* | BCF/VCF with `FORMAT/LRR` |
| `-o, --output` | path | *required* | Output BCF/VCF with corrected LRR |
| `--build` | str | auto-detect | Genome build: `GRCh37`, `GRCh38`, `T2T-CHM13` (aliases: `hg19`, `hg38`, `hs1`) |
| `--k` | int | auto (Marchenko-Pastur) | Number of batch PCs to remove |
| `--no-complexity-filter` | flag | `False` | Skip centromere / segdup / MHC exclusion |
| `--max-lrr-sd` | float | `0.35` | Max LRR-SD for HQ sample classification |
| `--min-sample-call-rate` | float | `0.97` | Min call rate for HQ samples |
| `--min-marker-call-rate` | float | `0.95` | Min marker call rate for SVD |
| `--min-var` | float | `0.001` | Min LRR variance for markers |
| `--max-var` | float | `None` | Max LRR variance for markers |
| `--backend` | str | `rsvd` | SVD backend: `rsvd` (scikit-learn) or `fbpca` |
| `--variant-qc` | path | `None` | Path to upstream `collated_variant_qc.tsv`; markers failing cross-ancestry call rate or HWE are excluded before decomposition |
| `--config` | path | `None` | YAML QC config file (see [QC Configuration](#qc-configuration)) |
| `-v, --verbose` | flag | `False` | Enable debug logging |

### `associate`

Test each marker's LRR against a phenotype.

```
array-lrr-gwas associate INPUT --phenotype PHENO -o OUTPUT [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `INPUT` | path | *required* | BCF/VCF with `FORMAT/LRR` (typically the corrected output) |
| `--phenotype` | path | *required* | Tab-separated file (see [Input Formats](#input-formats)) |
| `-o, --output` | path | *required* | Output TSV of per-marker association results |
| `--method` | str | `lmm` | `lmm` (default), `ols`, or `logistic` |
| `--sample-sheet` | path | `None` | `compiled_sample_sheet.tsv` for ancestry PCs |
| `--n-pcs` | int | `20` | Number of PCs to include as covariates |
| `--genotype-bcf` | path | `INPUT` | BCF/VCF for GRM computation (if different from input) |
| `--variant-qc` | path | `None` | Path to upstream `collated_variant_qc.tsv`; variants failing call rate/HWE/MAF are excluded before GRM |
| `--min-maf` | float | `0.01` | Min MAF for genotypes used in GRM |
| `--min-gt-call-rate` | float | `0.90` | Min call rate for genotypes used in GRM |
| `--no-ld-prune` | flag | `False` | Disable LD pruning of GRM markers |
| `--ld-window-bp` | int | `1000000` | LD-pruning window size in base pairs |
| `--ld-r2-thresh` | float | `0.2` | r² threshold for LD pruning |
| `--ld-backend` | str | `plink2` | LD-pruning backend: `plink2` (default, fast) or `numpy` (fallback) |
| `--config` | path | `None` | YAML config file used to read `upstream_qc.variant_qc_path` when `--variant-qc` is not provided |
| `-v, --verbose` | flag | `False` | Enable debug logging |

### `segment`

Define CNV-associated genomic intervals from association results.

```
array-lrr-gwas segment INPUT -o OUTPUT [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `INPUT` | path | *required* | Association TSV from `associate` |
| `-o, --output` | path | *required* | Output BED file |
| `--strategy` | str | `hmm` | `hmm` (two-state HMM) or `threshold` |
| `--p-threshold` | float | `5e-8` | Significance threshold (threshold strategy only) |
| `--max-gap` | int | `1000000` | Max gap (bp) to merge segments (threshold strategy only) |
| `--min-markers` | int | `1` | Min markers per segment |
| `--null-rate` | float | `ln(10)` | Null-state exponential rate (HMM only) |
| `--signal-rate` | float | `0.1` | Associated-state exponential rate (HMM only) |
| `--prior-assoc` | float | `0.001` | Prior probability of associated state (HMM only) |
| `--transition-prob` | float | `0.0001` | Per-marker state transition probability (HMM only) |
| `-v, --verbose` | flag | `False` | Enable debug logging |

---

## QC Configuration

Sample and marker QC thresholds can be set via a YAML configuration file
passed with `--config` to `correct`. The same config can be passed to
`associate` to provide `upstream_qc.variant_qc_path` for GRM marker filtering.
CLI flags take precedence over YAML values, which take precedence over
built-in defaults.

### Default Configuration

```yaml
sample_qc:
  max_lrr_sd: 0.35            # Max LRR SD for HQ classification
  min_call_rate: 0.97          # Min call rate for HQ classification

marker_qc:
  min_call_rate: 0.95          # Min marker call rate
  min_var: 0.001               # Min LRR variance
  max_var: null                # No upper limit

correction:
  k: null                      # Auto via Marchenko-Pastur
  backend: rsvd                # scikit-learn randomised SVD
  no_complexity_filter: false   # Apply centromere/segdup exclusion

upstream_qc:
  variant_qc_path: null        # Optional path to collated_variant_qc.tsv
```

### Stricter Thresholds Example

```yaml
sample_qc:
  max_lrr_sd: 0.30
  min_call_rate: 0.98

marker_qc:
  min_call_rate: 0.98
  min_var: 0.002
  max_var: 5.0

correction:
  k: 5
  backend: rsvd
```

---

## Input Formats

### BCF/VCF with FORMAT/LRR

The primary input is a BCF (or VCF) file produced by the upstream
[`jlanej/illumina_idat_processing`](https://github.com/jlanej/illumina_idat_processing)
pipeline.  Required FORMAT fields:

| Field | Type | Description |
|-------|------|-------------|
| `GT` | String | Genotype (`0/0`, `0/1`, `1/1`, `./.`) |
| `LRR` | Float | Log R Ratio |
| `BAF` | Float | B Allele Frequency (used upstream, not by this package) |

### Phenotype TSV

Tab-separated file with a header row.  The first column must be the sample
identifier; the second column is the phenotype (continuous or binary 0/1).
Additional columns are treated as covariates.  Columns **must** be separated
by literal tab characters (`\t`).

| sample\_id | phenotype | age | sex |
|------------|-----------|-----|-----|
| SAMPLE\_001 | 0.52 | 45 | 1 |
| SAMPLE\_002 | -0.13 | 62 | 0 |
| SAMPLE\_003 | 1.00 | 38 | 1 |

### Compiled Sample Sheet

A tab-separated file from the upstream pipeline containing sample QC metrics
and ancestry PCs (`PC1`–`PC20`).  Passed via `--sample-sheet` to the
`associate` command.  See [docs/upstream_qc_formats.md](docs/upstream_qc_formats.md)
for the full specification.

---

## Output Formats

### Corrected BCF/VCF (`correct`)

Same structure as the input BCF/VCF with the `FORMAT/LRR` values replaced by
batch-corrected values.  A `batch_lrr_correction` header line records the
parameters used (k, backend, n\_hq\_samples, n\_markers\_used, singular
values, timestamp).

### Association TSV (`associate`)

| Column | Type | Description |
|--------|------|-------------|
| `chrom` | str | Chromosome identifier |
| `pos` | int | 1-based position |
| `variant_id` | str | Variant ID or `chrom:pos` fallback |
| `beta` | float | Effect size (LRR → phenotype) |
| `se` | float | Standard error |
| `stat` | float | Test statistic (t or z) |
| `p_value` | float | Two-sided p-value |
| `n_samples` | int | Effective sample size |
| `method` | str | `lmm`, `ols`, or `logistic` |

### Segmentation BED (`segment`)

BED format (0-based, half-open) with additional columns:

| Column | Type | Description |
|--------|------|-------------|
| `chrom` | str | Chromosome |
| `start` | int | 0-based start |
| `end` | int | 0-based exclusive end |
| `name` | str | `region_1`, `region_2`, … |
| `n_markers` | int | Markers in segment |
| `min_p` | float | Minimum p-value |
| `mean_beta` | float | Mean effect size |
| `max_abs_stat` | float | Max \|test statistic\| |
| `method` | str | Association method |

---

## Python API

The package exports a public API for programmatic use:

```python
import numpy as np
from array_lrr_gwas import (
    read_genotypes, compute_grm,
    correct_lrr, run_association, segment_associations,
    read_sample_sheet, align_samples,
    detect_build, get_exclusion_regions,
    load_qc_config,
)
from array_lrr_gwas.io_vcf import read_lrr, write_corrected

# --- Load LRR from BCF ---
lrr, samples, variants = read_lrr("input.bcf")
positions = np.array([v["pos"] for v in variants])
chromosomes = np.array([v["chrom"] for v in variants])

# --- Batch-effect correction ---
corrected, info = correct_lrr(lrr, positions, chromosomes, k=None)
# info["k"], info["singular_values"], info["n_hq_samples"], ...

# --- Write corrected BCF ---
write_corrected("corrected.bcf", corrected, samples, variants, info,
                path_template="input.bcf")

# --- Load phenotype and covariates ---
phenotype = ...  # np.ndarray, shape (n_samples,)

sheet_ids, covariates, cov_names = read_sample_sheet(
    "compiled_sample_sheet.tsv", n_pcs=20
)
covariates = align_samples(samples, sheet_ids, covariates)

# --- Compute GRM ---
dosage, gt_samples, gt_variants = read_genotypes("genotypes.bcf")
grm = compute_grm(dosage)

# --- Run association ---
result = run_association(
    corrected, phenotype, variants,
    covariates=covariates, method="lmm", grm=grm,
)

# --- Segment ---
segments = segment_associations(result.to_records(), strategy="hmm")
segments.write_bed("regions.bed")
```

---

## Methodology

### Batch-Effect Correction

The LRR matrix **X** (n\_markers × n\_samples) is modelled as:

> **X = S + B + ε**

where **S** is true biological signal, **B** ≈ **U diag(s) Vᵀ** is a
low-rank batch-effect component, and **ε** is measurement noise.

**Procedure:**

1. **Classify samples:** Samples with `LRR_SD ≤ 0.35` and
   `call_rate ≥ 0.97` are high-quality (HQ); the rest are low-quality (LQ).
2. **Subset markers:** Remove markers with poor call rate, near-zero or
   extreme variance, or residing in problematic regions (centromeres,
   segmental duplications, MHC, immunoglobulin loci).
3. **Decompose:** Perform randomised SVD on the HQ sub-matrix of passing
   markers to extract the top *k* principal components.
4. **Select k:** The Marchenko-Pastur (MP) threshold from random-matrix
   theory determines how many components exceed the noise floor.  The MP
   upper edge is:

   > λ₊ = σ² (1 + √γ)²

   where σ² is estimated from the median squared singular value and
   γ = n\_markers / n\_samples.  Components with eigenvalue above λ₊ are
   retained.

5. **Extrapolate:** LQ sample PCs are estimated by projecting onto the
   HQ-derived marker loadings: **V\_LQ = (1/s) Uᵀ X\_LQ**.
6. **Residualise:** Subtract the batch component from **all** markers (not
   just the subset): **X\_corrected = X − U diag(s) Vᵀ**.

### Association Analysis

Three methods are available:

#### Linear Mixed Model (LMM, default)

Accounts for cryptic relatedness via a Genetic Relationship Matrix (GRM)
random effect, following the FaST-LMM / EMMA approach:

> phenotype ~ LRR\_i + covariates + (1 | GRM)

1. **Eigendecompose** the GRM: **K = U Λ Uᵀ**.
2. **Rotate** phenotype, covariates, and LRR into the GRM eigenbasis.
3. **Estimate δ = σ²\_e / σ²\_g** under the null (no marker) via profile
   REML.
4. **Per-marker WLS** in rotated space with weights
   **w\_i = 1 / (λ\_i + δ)**.

The GRM is computed using the Yang et al. (2011) standardised estimator from
genotype dosages (FORMAT/GT), with markers filtered at MAF ≥ 0.01.

Before GRM computation, markers are LD-pruned by default (r² threshold 0.2
within a 1 Mb window) so that highly linked regions do not dominate the GRM
eigenstructure. Use `--no-ld-prune` to disable pruning, or tune
`--ld-r2-thresh` / `--ld-window-bp`. The default backend is `plink2`, with
automatic fallback to `numpy` if `plink2` is unavailable.

For scalability, the LRR × eigenvector rotation is processed in chunks of
10,000 markers to bound peak memory.

#### Ordinary Least Squares (OLS)

Standard regression without relatedness correction.  Uses the
Frisch-Waugh-Lovell theorem for efficient per-marker residualisation.
Suitable when samples are unrelated or relatedness has been removed by
prior filtering.

#### Logistic Regression

Iteratively Re-weighted Least Squares (IRLS) for binary phenotypes (0/1).
**Does not incorporate a GRM random effect** — cryptic relatedness is not
accounted for.  Users should pre-filter related individuals before using
this method.

### Segmentation

#### Two-State HMM (default)

Observations are **−log₁₀(p)** per marker.  Under H₀, p-values are
Uniform(0, 1), so −log₁₀(p) follows Exponential(rate = ln 10 ≈ 2.303).
Under H₁ (CNV-associated), a smaller rate (default 0.1) expects −log₁₀(p)
around 10.

A symmetric transition matrix encodes the prior that associated regions are
rare.  Viterbi decoding yields the most likely state sequence; contiguous
runs of the associated state become genomic intervals.

#### Threshold-and-Merge

1. Flag markers below a p-value threshold (default 5 × 10⁻⁸).
2. Merge contiguous flagged runs.
3. Merge segments separated by ≤ `max_gap` bp (default 1 Mb).

---

## Limitations and Caveats

This section describes known limitations.  We believe transparent reporting
of these issues is essential for responsible use.

### Batch-Effect Correction

- **Low-rank assumption.**  The correction assumes batch effects are
  approximately low-rank.  If batch effects are non-linear or interact
  with biological signal, the SVD-based approach may be incomplete.
- **Signal leakage risk.**  If true CNV signal correlates with batch
  structure (e.g. cases processed on different scanner plates), the
  correction may remove some genuine biological signal along with the batch
  effect.  We mitigate this by estimating the decomposition from HQ
  samples only, but the risk cannot be fully eliminated.  Users should
  evaluate whether phenotype and batch variables are confounded.
- **LQ sample projection.**  Low-quality samples receive batch PCs
  estimated by projection onto HQ loadings.  This assumes the batch
  subspace is shared between HQ and LQ samples, which may not hold if LQ
  samples have systematically different artefact profiles.
- **Marchenko-Pastur threshold.**  The automatic *k* selection is
  conservative and may under- or over-estimate the true number of batch
  components, particularly for small sample sizes or heavy-tailed noise.

### Association Analysis

- **Null-model δ reuse.**  The LMM estimates the variance-component ratio
  δ under the null model once and reuses it for all markers.  This fast
  approximation is slightly conservative for very strong signals.
- **Binary traits in LMM.**  The LMM is designed for continuous traits.
  When applied to binary phenotypes as a linear approximation, it may
  inflate Type I error if the case fraction is highly unbalanced
  (< 10% or > 90%).  A warning is emitted in this situation.
- **Logistic regression ignores relatedness.**  The logistic model does
  not incorporate a GRM random effect.  Users must pre-filter related
  individuals (e.g. one from each pair with kinship > 0.125) to control
  false positives.
- **Per-marker missing data.**  Missing LRR values trigger a slower
  per-marker complete-case analysis loop.  Minimising upstream missingness
  improves performance and avoids potential bias if data are not missing
  completely at random (MCAR).
- **Convergence failures (logistic).**  IRLS has a fixed iteration cap
  (25 iterations, tol = 1e-6).  Non-convergent markers silently produce
  NaN beta and p = 1.  This typically indicates separation or extreme
  class imbalance for that marker.
- **LRR is a proxy, not a direct measure.**  LRR reflects total signal
  intensity, not true copy number.  It is influenced by GC content,
  probe design, and DNA quality.  Significant associations should be
  validated with orthogonal methods (e.g. WGS, digital PCR).

### Segmentation

- **Independence assumption.**  The HMM treats markers as conditionally
  independent given the hidden state.  It does not explicitly model
  linkage disequilibrium or probe spacing density, which varies across
  the genome.
- **Fixed emission parameters.**  The null and signal emission rates are
  fixed hyperparameters, not learned from data.  Mis-specification can
  lead to over- or under-segmentation.
- **Single-pass Viterbi.**  Decoding is a single forward pass with no
  iterative re-estimation (no Baum-Welch).  This is fast but may be
  suboptimal if the true emission parameters differ substantially from
  defaults.
- **p = 0 floor.**  p-values of exactly zero are clipped to 10⁻³⁰⁰ to
  avoid −∞ in the log transform, which may cause over-smoothing for
  extremely significant markers.

### General

- **Memory.**  The full LRR matrix must fit in RAM for the correction
  step.  For very large cohorts (> 500k samples), consider subsetting or
  using chunked I/O approaches externally.
- **Single-threaded.**  The current implementation is single-threaded.
  Parallelism is best achieved by running separate jobs per chromosome.
- **Array platforms.**  Tested with Illumina arrays.  Affymetrix LRR
  values have different scaling and noise characteristics; default
  thresholds may need adjustment.

---

## Upstream Dependencies

This package expects input from the
[`jlanej/illumina_idat_processing`](https://github.com/jlanej/illumina_idat_processing)
pipeline, which converts Illumina IDAT files to BCF/VCF via
`bcftools +gtc2vcf` and produces a compiled sample sheet with QC metrics and
ancestry PCs.  See [docs/upstream_qc_formats.md](docs/upstream_qc_formats.md)
for the full format specification.

---

## Development

```bash
# Install with dev dependencies
pip install -e '.[dev]'

# Run tests
pytest tests/ -v --tb=short

# Build Docker image locally
docker build -t array-lrr-gwas .
docker run --rm array-lrr-gwas --help
```

### Project Structure

```
array_lrr_gwas/
├── __init__.py          # Public API exports
├── cli.py               # Command-line interface (correct, associate, segment)
├── subsetting.py        # Marker QC filters (call-rate, variance, complexity)
├── decomposition.py     # Randomised SVD (rsvd / fbpca backends)
├── correction.py        # End-to-end batch correction pipeline
├── select_k.py          # Component selection (Marchenko-Pastur, elbow)
├── genome_build.py      # Build detection + exclusion regions
├── qc_config.py         # YAML configuration with override hierarchy
├── io_vcf.py            # BCF/VCF I/O for LRR
├── genotypes.py         # Genotype extraction (FORMAT/GT → dosage)
├── grm.py               # Genetic Relationship Matrix (Yang et al. 2011)
├── sample_sheet.py      # Compiled sample sheet parser
├── association.py        # LMM / OLS / logistic association scans
└── segmentation.py       # HMM / threshold-merge segmentation
docs/
├── association_engine_design.md
├── segmentation_design.md
└── upstream_qc_formats.md
tests/
├── conftest.py          # Shared fixtures (synthetic LRR, test BCF)
├── test_*.py            # Per-module test suites
└── data/
    └── test.bcf         # Bundled test BCF
```

---

## References

- Anderson, C. A. et al. *Data quality control in genetic case-control
  association studies.* Nature Protocols **5**, 1564–1573 (2010).
- Halko, N., Martinsson, P. G. & Tropp, J. A. *Finding structure with
  randomness: Probabilistic algorithms for constructing approximate matrix
  decompositions.* SIAM Review **53**, 217–288 (2011).
- Kang, H. M. et al. *Variance component model to account for sample
  structure in genome-wide association studies.* Nature Genetics **42**,
  348–354 (2010). [EMMA]
- Lippert, C. et al. *FaST linear mixed models for genome-wide association
  studies.* Nature Methods **8**, 833–835 (2011). [FaST-LMM]
- Marchenko, V. A. & Pastur, L. A. *Distribution of eigenvalues for some
  sets of random matrices.* Mathematics of the USSR-Sbornik **1**, 457–483
  (1967).
- Yang, J. et al. *GCTA: A tool for genome-wide complex trait analysis.*
  American Journal of Human Genetics **88**, 76–82 (2011).

---

## License

This project is licensed under the
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).
See [LICENSE](LICENSE) for the full text.
