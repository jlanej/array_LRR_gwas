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
  - [Running Directly on illumina_idat_processing Output](#running-directly-on-illumina_idat_processing-output)
- [Running on HPC with Apptainer](#running-on-hpc-with-apptainer)
- [CLI Reference](#cli-reference)
  - [correct](#correct)
  - [associate](#associate)
  - [segment](#segment)
  - [report](#report)
  - [diagnostic-report](#diagnostic-report)
- [Sample Exclusion Strategies](#sample-exclusion-strategies)
  - [RSVD Correction Stage](#rsvd-correction-stage)
  - [Association Stage](#association-stage)
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
│  ② Exclude markers: INTENSITY_ONLY, monomorphic LRR                 │
│  ③ Compute GRM from genotypes (for LMM)                            │
│  ④ LMM: spectral decomposition → REML δ → per-marker WLS          │
│     OR OLS / logistic regression (no GRM)                           │
│  ⑤ Write per-marker results TSV                                     │
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
# --genotype-bcf is optional: omit it when the input BCF already contains GT.
array-lrr-gwas associate corrected.bcf \
    --phenotype pheno.tsv \
    --sample-sheet compiled_sample_sheet.tsv \
    --method lmm \
    -o results.tsv -v

# Step 3 — Segment into CNV regions
array-lrr-gwas segment results.tsv -o regions.bed -v

# Step 4 — Generate a publication-quality interactive HTML summary report
# (Manhattan + QQ + regional plots, lambda_GC, top-hit tables, and UCSC
# gene annotations) covering the autosomal scan and all non-autosomal
# modes (chrX/chrY/chrM/MT). Gene tracks are auto-downloaded from UCSC
# on first use and cached locally.
array-lrr-gwas report \
    --autosomal results.tsv \
    --x-with-sex-covariate results.x_with_sex_covariate.tsv \
    --x-male-only results.x_male_only.tsv \
    --x-female-only results.x_female_only.tsv \
    --y-male-only results.y_male_only.tsv \
    --mt-with-sex-covariate results.mt_with_sex_covariate.tsv \
    --mt-male-only results.mt_male_only.tsv \
    --mt-female-only results.mt_female_only.tsv \
    --build GRCh38 --gene-window-kb 500 \
    -o gwas_report.html

# (Alternatively, pass --report gwas_report.html to `associate` to have
#  the report generated automatically after the scan completes.)
```

**Best-practice workflow with upstream QC filtering:**

```bash
# Step 1 — Remove batch effects with upstream variant QC
array-lrr-gwas correct input.bcf -o corrected.bcf --build GRCh38 \
    --variant-qc collated_variant_qc.tsv -v

# Step 2 — Run LMM association with QC-filtered GRM
# --genotype-bcf is only needed when GT lives in a *separate* file from the
# LRR input.  When the input BCF contains both GT and LRR (the typical case
# for illumina_idat_processing output), it can be safely omitted.
array-lrr-gwas associate corrected.bcf \
    --phenotype pheno.tsv \
    --sample-sheet compiled_sample_sheet.tsv \
    --variant-qc collated_variant_qc.tsv \
    --method lmm \
    -o results.tsv -v

# Step 3 — Segment into CNV regions
array-lrr-gwas segment results.tsv -o regions.bed -v
```

### Running Directly on illumina\_idat\_processing Output

The examples below use the **exact file paths** produced by a typical run
of [`jlanej/illumina_idat_processing`](https://github.com/jlanej/illumina_idat_processing).
Adjust `PIPELINE_DIR` to point at your own pipeline run directory.

```bash
PIPELINE_DIR=/data/project/my_run

# Key files produced by illumina_idat_processing:
#   ${PIPELINE_DIR}/stage2/vcf/stage2_reclustered.bcf   — BCF with GT, LRR, BAF
#   ${PIPELINE_DIR}/compiled_sample_sheet.tsv            — sample QC + ancestry PCs
#   ${PIPELINE_DIR}/ancestry_stratified_qc/collated_variant_qc.tsv — per-variant QC flags
# If the upstream run used --skip-stage2, use:
#   ${PIPELINE_DIR}/stage1/vcf/stage1_initial.bcf

INPUT_BCF="${PIPELINE_DIR}/stage2/vcf/stage2_reclustered.bcf"
SAMPLE_SHEET="${PIPELINE_DIR}/compiled_sample_sheet.tsv"
VARIANT_QC="${PIPELINE_DIR}/ancestry_stratified_qc/collated_variant_qc.tsv"
PHENO="${PIPELINE_DIR}/phenotype.tsv"   # user-provided phenotype file
OUTDIR="${PIPELINE_DIR}/lrr_gwas"
mkdir -p "${OUTDIR}"

# Step 1 — Batch-effect correction (build is auto-detected from BCF contigs)
# HQ/LQ sample classification here is derived directly from BCF LRR values.
# --sample-sheet is only needed at association stage for sample exclusions
# and non-autosomal (sex chromosome + mitochondrial) analyses.
array-lrr-gwas correct "${INPUT_BCF}" \
    -o "${OUTDIR}/corrected.bcf" \
    --variant-qc "${VARIANT_QC}" \
    -v

# Step 2 — Association (LMM)
# No --genotype-bcf needed: the reclustered BCF already contains FORMAT/GT.
array-lrr-gwas associate "${OUTDIR}/corrected.bcf" \
    --phenotype "${PHENO}" \
    --sample-sheet "${SAMPLE_SHEET}" \
    --variant-qc "${VARIANT_QC}" \
    --method lmm \
    -o "${OUTDIR}/association_results.tsv" \
    -v

# Step 3 — Segmentation
array-lrr-gwas segment "${OUTDIR}/association_results.tsv" \
    -o "${OUTDIR}/cnv_regions.bed" \
    -v
```

> **Tip:** If your genotypes live in a *different* BCF (e.g. an
> imputed file), pass it explicitly with
> `--genotype-bcf /path/to/imputed.bcf`.  Otherwise, simply omit the
> flag.

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

# illumina_idat_processing output paths
PIPELINE_DIR=/data/project/my_run
INPUT_BCF="${PIPELINE_DIR}/stage2/vcf/stage2_reclustered.bcf"
VARIANT_QC="${PIPELINE_DIR}/ancestry_stratified_qc/collated_variant_qc.tsv"
SAMPLE_SHEET="${PIPELINE_DIR}/compiled_sample_sheet.tsv"
PHENO=/data/project/phenotype.tsv
OUTDIR=/data/project/results
mkdir -p "${OUTDIR}"

# Step 1: Batch-effect correction
apptainer run --bind /data "${SIF}" correct \
    "${INPUT_BCF}" \
    -o "${OUTDIR}/corrected.bcf" \
    --variant-qc "${VARIANT_QC}" \
    -v

# Step 2: Association (LMM with GRM; phenotype and covariates from phenotype file)
# LD pruning is enabled by default for GRM computation.
# --genotype-bcf is omitted: the reclustered BCF already contains FORMAT/GT.
apptainer run --bind /data "${SIF}" associate \
    "${OUTDIR}/corrected.bcf" \
    --phenotype "${PHENO}" \
    --sample-sheet "${SAMPLE_SHEET}" \
    --variant-qc "${VARIANT_QC}" \
    --method lmm \
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
| `--k` | int | auto (Marchenko-Pastur) | Number of batch PCs to remove from LRR |
| `--n-components` | int | auto (5% of HQ samples) | Number of PCs to compute in pilot decomposition for auto-`k` |
| `--no-complexity-filter` | flag | `False` | Skip centromere / segdup / MHC exclusion |
| `--max-lrr-sd` | float | `0.35` | Max LRR-SD for HQ sample classification |
| `--min-sample-call-rate` | float | `0.97` | Min call rate for HQ samples |
| `--min-marker-call-rate` | float | `0.95` | Min marker call rate for SVD |
| `--min-var` | float | `0.001` | Min LRR variance for markers |
| `--max-var` | float | `None` | Max LRR variance for markers |
| `--backend` | str | `rsvd` | SVD backend: `rsvd` (scikit-learn) or `fbpca` |
| `--svd-output-prefix` | path | `<OUTPUT>.svd` | Prefix for SVD text summaries (`.sample_pcs.tsv`, `.singular_values.tsv`) |
| `--write-loadings` | flag | `False` | Also write marker loadings to `<prefix>.loadings.tsv` |
| `--no-interactive-report` | flag | `False` | Skip generation of the interactive HTML diagnostic report (scree plot, 3-D PC scatter, UMAP projection) |
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
| `--sample-sheet` | path | `None` | `compiled_sample_sheet.tsv` used to derive HQ samples (when `--hq-samples` is omitted) and for sex-chromosome modes |
| `--phenotype-col` | str | first non-`sample_id` column | Column in `--phenotype` to use as the phenotype |
| `--covariate-cols` | str(s) | all remaining non-`sample_id` columns | Columns in `--phenotype` to use as fixed-effect covariates |
| `--hq-samples` | path | `None` | Optional HQ sample list; analyzes only `valid phenotype ∩ HQ` samples (drops LQ). When omitted but `--sample-sheet` is provided, HQ samples are derived from the sheet using `--max-lrr-sd` and `--min-sample-call-rate` |
| `--max-lrr-sd` | float | `0.35` | Max per-sample LRR SD for HQ classification (used when deriving HQ from sample sheet) |
| `--min-sample-call-rate` | float | `0.97` | Min per-sample call rate for HQ classification (used when deriving HQ from sample sheet) |
| `--no-honor-precomputed` | flag | `False` | Disable honoring pre-computed exclusion columns (`pre_pca_excluded`, `excluded_relatedness`, `excluded_het_outlier`) from sample sheet. See [Sample Exclusion Strategies](#sample-exclusion-strategies) |
| `--no-exclude-baf-sd` | flag | `False` | Disable BAF SD exclusion (samples with `baf_sd > --max-baf-sd` are excluded by default as a contamination proxy) |
| `--max-baf-sd` | float | `0.15` | BAF SD threshold for sample exclusion. Higher BAF SD suggests DNA contamination (Marees et al. 2018) |
| `--no-exclude-sex-discordant` | flag | `False` | Disable sex-discordance exclusion (`sex_status == "DISCORDANT"` excluded by default; Anderson et al. 2010) |
| `--no-exclude-extreme-inbreeding` | flag | `False` | Disable extreme inbreeding coefficient exclusion (\|F\| > threshold excluded by default) |
| `--max-abs-inbreeding-f` | float | `0.15` | Inbreeding coefficient threshold. Samples with \|F\| above this are excluded (Anderson et al. 2010) |
| `--genotype-bcf` | path | `INPUT` | **Optional.** BCF/VCF for GRM computation. Defaults to the input file when omitted (requires `FORMAT/GT`). Only needed when genotypes live in a separate file from the LRR input. |
| `--variant-qc` | path | `None` | Path to upstream `collated_variant_qc.tsv`; markers failing call rate/HWE/MAF are excluded from autosomal GRM construction. For X-GRM, chrX-specific upstream columns (`all_ancestries_chrX_female_hwe_pass`, `all_ancestries_chrX_call_rate_pass`) are used when present (with autosomal fallback). Per-marker QC flags are propagated to the output TSV for post-hoc filtering; LRR association markers are not pre-filtered by these flags. |
| `--min-maf` | float | `0.01` | Min MAF for genotypes used in GRM |
| `--min-gt-call-rate` | float | `0.90` | Min call rate for genotypes used in GRM |
| `--no-ld-prune` | flag | `False` | Disable LD pruning of GRM markers |
| `--ld-window-bp` | int | `1000000` | LD-pruning window size in base pairs |
| `--ld-r2-thresh` | float | `0.2` | r² threshold for LD pruning |
| `--ld-backend` | str | `plink2` | LD-pruning backend: `plink2` (default, **required** — install from [cog-genomics.org](https://www.cog-genomics.org/plink/2.0/)) or `numpy` (explicit fallback). If plink2 is not on PATH and `plink2` is selected, the pipeline exits with an error. |
| `--no-exclude-intensity-only` | flag | `False` | Retain INTENSITY_ONLY markers in association (excluded by default because they lack GT) |
| `--no-exclude-monomorphic-lrr` | flag | `False` | Retain markers with zero LRR variance in association |
| `--sex-chr-mode` | str(s) | all modes when `--sample-sheet` provided | Non-autosomal scans (sex chromosomes + chrM/MT). When `--sample-sheet` is provided, all seven modes run by default: `x_with_sex_covariate`, `x_male_only`, `x_female_only`, `y_male_only`, `mt_with_sex_covariate`, `mt_male_only`, `mt_female_only`. Pass `--sex-chr-mode` with no arguments to skip all non-autosomal analyses. Requires `--sample-sheet` with `predicted_sex` column (1=male, 2=female). Each mode writes a separate TSV (e.g. `results.x_male_only.tsv`, `results.mt_with_sex_covariate.tsv`). **Constant covariates** (zero variance among the analysed stratum, e.g. a sex covariate in `x_male_only`) are automatically dropped with a warning. When `--method lmm` is used, chrX modes compute a dedicated X-chromosome GRM (X-GRM) with male 0/2 dosage coding, PAR exclusion, and sex-aware standardisation; chrY and chrM/MT modes reuse the autosomal GRM subsetted to the relevant stratum (mtDNA is maternally inherited and does not recombine, so the nuclear GRM is used only as a practical adjuster for cohort structure and batch effects). Falls back to OLS with a warning when X-GRM computation fails. |
| `--build` | str | auto-detect | Genome build: `GRCh37`, `GRCh38`, `T2T-CHM13` (aliases: `hg19`, `hg38`, `hs1`). Auto-detected from the input file when possible. Used for PAR region exclusion in X-GRM computation when `--sex-chr-mode` is enabled. |
| `--config` | path | `None` | YAML config file; reads `upstream_qc.variant_qc_path`, `sample_qc`, `association_qc`, and `association_marker_qc` settings |
| `--audit-dir` | path | `None` | Directory for structured audit trail (per-stage TSV, JSON summary with included/excluded fractions). Enables full provenance tracking. |
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

### `report`

Generate a single self-contained, interactive HTML summary report
from one or more association TSVs (autosomal scan plus any of the
seven non-autosomal modes: four sex-chromosome modes and three
chrM/MT modes).  The report collates, for each mode:

* An interactive Manhattan plot (non-significant points downsampled
  for file size; every variant with *p* < 10⁻⁵ is always kept) with
  genome-wide (dashed red, *p* < 5×10⁻⁸) and suggestive (dotted
  orange, *p* < 10⁻⁵) thresholds and gene labels annotating the
  nearest gene at each genome-wide-significant locus.
* A QQ plot with a 95 % confidence envelope under the null and the
  genomic inflation factor λ<sub>GC</sub>.
* Regional locus-zoom-style plots for the top loci, including a
  gene track below the association panel (strand-aware).
* A top-hits table with effect sizes, *p*-values, nearest gene,
  distance to nearest gene, and all genes falling inside a
  configurable window.
* A methods narrative and figure legends appropriate for a
  manuscript's supplementary materials.

Gene annotations are pulled from the canonical UCSC gene tracks —
`refGene` for GRCh37/GRCh38 and `ncbiRefSeq` for T2T-CHM13 (hs1) —
and are auto-downloaded on first use and cached on disk.  The
report is a single HTML file that uses Plotly from the CDN for
interactive rendering.

```
array-lrr-gwas report [INPUTS] -o OUTPUT.html [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--autosomal` | path | — | Autosomal association TSV |
| `--x-with-sex-covariate` | path | — | chrX full-cohort TSV |
| `--x-male-only` | path | — | chrX males-only TSV |
| `--x-female-only` | path | — | chrX females-only TSV |
| `--y-male-only` | path | — | chrY males-only TSV |
| `--mt-with-sex-covariate` | path | — | chrM/MT full-cohort TSV (sex as covariate) |
| `--mt-male-only` | path | — | chrM/MT males-only TSV |
| `--mt-female-only` | path | — | chrM/MT females-only (maternal-lineage) TSV |
| `-o, --output` | path | *required* | Output HTML file |
| `--build` | str | — | Genome build (`GRCh37`/`GRCh38`/`T2T-CHM13` or aliases `hg19`/`hg38`/`hs1`).  Required for gene annotation. |
| `--gene-window-kb` | int | `500` | Half-window (kb) for nearby-gene annotation and regional plots |
| `--top-n` | int | `10` | Number of top hits to list per mode |
| `--cache-dir` | path | `~/.cache/array_lrr_gwas` | UCSC download cache (also honours `$ARRAY_LRR_GWAS_CACHE`) |
| `--title` | str | *see default* | Report title |
| `--no-gene-annotation` | flag | `False` | Skip all UCSC downloads |
| `--top-hits-tsv-dir` | path | — | Also write `top_hits.<mode>.tsv` files here |
| `-v, --verbose` | flag | `False` | Enable debug logging |

The `associate` subcommand additionally accepts `--report
PATH.html` (plus `--report-gene-window-kb`, `--report-cache-dir`,
and `--no-report-gene-annotation`) to auto-generate the report
over the autosomal scan and every non-autosomal (chrX/Y/M) mode
that ran.  The report now opens with a **combined genome-wide
summary Manhattan plot** that overlays the autosomal scan, chrX
(`x_with_sex_covariate`), chrY (`y_male_only`), and chrM/MT
(`mt_with_sex_covariate`) results on a single axis for a
one-glance view across the whole genome.

---

### `diagnostic-report`

Regenerate the interactive LRR correction diagnostic HTML report **from
sidecar files** written by `correct` — without reloading the original BCF/VCF
or LRR matrix.  This is useful when you want to:

* Quickly tweak the report aesthetics or colour overlays with a different
  `--sample-sheet`.
* Re-run report generation as part of a downstream workflow after the
  `correct` step has already finished.
* Archive a fresh copy of the report alongside analysis notebooks.

**Prerequisites:** The following sidecar files must exist at `--svd-prefix`
(all written automatically by `correct`):

| File | Contents |
|------|---------|
| `<prefix>.correction_info.json` | k, n\_hq\_samples, n\_markers\_used, per-sample HQ mask |
| `<prefix>.sample_pcs.tsv` | Per-sample PC scores |
| `<prefix>.singular_values.tsv` | Singular values with used-for-correction flag |
| `<prefix>.sample_metrics.tsv` | Per-sample LRR\_SD, callrate, n\_markers\_used (pre/post) |
| `<prefix>.umap.tsv` *(optional)* | Pre-computed UMAP coordinates — loaded when present |

```
array-lrr-gwas diagnostic-report --svd-prefix PREFIX -o OUTPUT.html [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--svd-prefix` | path | *required* | SVD prefix used when running `correct` (matches `--svd-output-prefix`) |
| `-o, --output` | path | *required* | Output HTML file for the regenerated report |
| `--sample-sheet` | path | `None` | Compiled sample sheet TSV for colour overlays |
| `--illumina-sample-sheet` | path | `None` | Illumina-format SampleSheet.csv for additional overlays |
| `--illumina-sample-id-col` | str | `Sample_Group` | Column in the Illumina sheet whose values match sample IDs |
| `--skip-umap` | flag | `False` | Skip UMAP even if `<prefix>.umap.tsv` exists |
| `-v, --verbose` | flag | `False` | Enable debug logging |

**Example workflow:**

```bash
# Step 1: run correction (writes all sidecars automatically)
array-lrr-gwas correct input.bcf -o corrected.bcf \
    --sample-sheet compiled_sample_sheet.tsv

# Step 2 (later): regenerate the report, adding an Illumina sheet
array-lrr-gwas diagnostic-report \
    --svd-prefix corrected.bcf.svd \
    -o corrected.bcf.diagnostic_report_v2.html \
    --sample-sheet compiled_sample_sheet.tsv \
    --illumina-sample-sheet SampleSheet.csv
```

---

## Sample Exclusion Strategies

Rigorous sample exclusion is essential for both the RSVD batch-effect
correction and the downstream GWAS association analysis.  This section
documents the exclusion criteria applied at each stage, their scientific
rationale, and how to configure them.

All exclusion decisions are **logged with per-category counts** for full
reproducibility and provenance.

### RSVD Correction Stage

The `correct` sub-command classifies samples as high-quality (HQ) or
low-quality (LQ).  Only HQ samples contribute to the SVD decomposition;
LQ samples receive corrected values via projection.

| Criterion | Default Threshold | Rationale |
|-----------|-------------------|-----------|
| **Low call rate** | `call_rate < 0.97` | Samples with excessive genotype missingness introduce noise into the decomposition. Threshold follows upstream `illumina_idat_processing` defaults and GWAS QC guidance (Anderson et al. 2010 recommend ≥ 0.95–0.98). |
| **High LRR SD** | `lrr_sd > 0.35` | Per-sample LRR standard deviation is the primary noise metric. Noisy samples degrade the SVD signal-to-noise ratio. Threshold from Marees et al. 2018 and upstream pipeline. |

Both thresholds are user-configurable via `--max-lrr-sd`,
`--min-sample-call-rate`, or the `sample_qc` section in the YAML
`--config` file.

**Note:** Samples with missing phenotype values are *not* excluded at the
correction stage — the correction operates on the full LRR matrix so that
all samples receive batch-corrected values regardless of downstream
analysis inclusion.

### Association Stage

The `associate` sub-command applies a richer set of exclusion criteria when
a `--sample-sheet` is provided.  These are applied *in addition to* the
core HQ criteria (call rate + LRR SD) and the requirement for a valid
(non-missing) phenotype value.

#### Default Exclusions (always ON)

| Criterion | Column(s) | Default | Rationale |
|-----------|-----------|---------|-----------|
| **Low call rate** | `call_rate` | `< 0.97` | Same as correction stage. |
| **High LRR SD** | `lrr_sd` | `> 0.35` | Same as correction stage. |
| **Pre-PCA excluded** | `pre_pca_excluded` | `true` ⇒ exclude | Sample was excluded before ancestry PCA by upstream QC (e.g. failed genotype QC). Honors upstream pipeline decisions. |
| **Excluded relatedness** | `excluded_relatedness` | `true` ⇒ exclude | Sample removed as part of a related pair (typically up to 2nd degree, kinship > 0.0884). While the GRM/LMM corrects for moderate kinship, removing close relatives reduces bias in effect-size estimates and avoids overcounting families (UK Biobank, TOPMed, Broad Institute conventions). |
| **Het outlier** | `excluded_het_outlier` | `true` ⇒ exclude | Extreme heterozygosity outlier, suggesting potential DNA contamination or sample mix-up. |
| **High BAF SD** | `baf_sd` | `> 0.15` ⇒ exclude | B Allele Frequency standard deviation is a contamination proxy. Elevated values suggest DNA contamination (Marees et al. 2018). |
| **Sex discordant** | `sex_status` | `"DISCORDANT"` ⇒ exclude | Discordance between reported and inferred sex flags potential sample swaps (Anderson et al. 2010 Nat Protoc). |
| **Extreme inbreeding F** | `inbreeding_F` | `|F| > 0.15` ⇒ exclude | Extreme inbreeding coefficient values indicate extreme population structure (F > 0) or potential contamination (F < 0). Threshold from Anderson et al. 2010. |

All exclusion categories are enabled by default.  Each can be **disabled
individually** via CLI flags or the `association_qc` section in the YAML
config:

| CLI Flag | Config Key | Effect |
|----------|------------|--------|
| `--no-honor-precomputed` | `association_qc.honor_precomputed: false` | Ignore `pre_pca_excluded`, `excluded_relatedness`, `excluded_het_outlier` |
| `--no-exclude-baf-sd` | `association_qc.exclude_baf_sd: false` | Skip BAF SD filter |
| `--max-baf-sd 0.20` | `association_qc.max_baf_sd: 0.20` | Change BAF SD threshold |
| `--no-exclude-sex-discordant` | `association_qc.exclude_sex_discordant: false` | Skip sex-discordance filter |
| `--no-exclude-extreme-inbreeding` | `association_qc.exclude_extreme_inbreeding: false` | Skip inbreeding-F filter |
| `--max-abs-inbreeding-f 0.20` | `association_qc.max_abs_inbreeding_f: 0.20` | Change \|F\| threshold |

**Missing columns** for optional criteria are silently skipped (with an
info-level log message), so the pipeline works with minimal sample sheets
containing only `Sample_ID`, `call_rate`, and `lrr_sd`.

#### Usage Example (Automatic from illumina\_idat\_processing Output)

```bash
# Step 2: Associate with full exclusion strategy (all defaults applied)
# --genotype-bcf is omitted (input BCF contains GT).
array-lrr-gwas associate corrected.bcf \
    --phenotype phenotype.tsv \
    --sample-sheet compiled_sample_sheet.tsv \
    --variant-qc ancestry_stratified_qc/collated_variant_qc.tsv \
    --method lmm \
    -o results.tsv -v
```

The log output will include a detailed exclusion summary:

```
INFO: Sample exclusion summary (sheet total=5000): low_call_rate=23,
      high_lrr_sd=45, pre_pca_excluded=12, excluded_relatedness=187,
      excluded_het_outlier=3, high_baf_sd=7, sex_discordant=2,
      extreme_inbreeding_f=1, total_excluded=267, passing=4733
```

#### Trade-offs and Scientific Consensus

The GRM-based LMM corrects for moderate cryptic relatedness in the
random-effect component.  However, best-practice GWAS pipelines (UK
Biobank, TOPMed, Broad Institute, eMERGE) still recommend excluding close
relatives (up to 2nd degree) because:

1. The LMM assumes the GRM captures population structure — very close
   relatives (parent-child, full siblings) inflate the GRM diagonal and
   can bias variance-component estimates.
2. Removing one member of a related pair eliminates redundant phenotype
   information that inflates effective sample size.
3. Family-correlated environmental effects are not modelled by genomic
   relatedness alone.

By defaulting all exclusions to ON, the pipeline errs on the side of
conservative, robust inference.  Users who prefer to retain all samples
(e.g. for family-aware analyses) can disable exclusions with the `--no-*`
flags.

---

## Marker Exclusion Criteria

Marker (variant) exclusion is applied at both the correction and
association stages, with stage-specific filters reflecting the different
requirements of each analysis.  All filtering decisions are **logged with
per-step counts** and can be overridden via CLI flags or YAML
configuration.

### Correction Stage (RSVD Decomposition)

The `correct` sub-command filters markers before the SVD decomposition to
ensure stable, denoised batch-effect estimation.

| Filter | Default | Rationale |
|--------|---------|-----------|
| **Low call rate** | `call_rate < 0.95` | Markers with excessive missingness introduce noise into the decomposition (Anderson et al. 2010). |
| **Low LRR variance** | `variance < 0.001` | Near-constant markers are uninformative for SVD and add no signal for batch-effect estimation. |
| **High LRR variance** | `None` (disabled) | Optionally exclude markers with extreme variance (set `--max-var 1.0` to remove rare artefactual outliers). |
| **Non-autosomal** | Exclude chrX, chrY, chrMT | Sex-linked intensity signals would cause PCA to capture sex rather than technical batch effects. |
| **Complexity regions** | Centromeres, segdups | Markers in regions of low mappability or segmental duplication are unreliable (build-specific). |
| **Upstream variant QC** | Call rate + HWE (not MAF) | When `--variant-qc` is provided, markers failing `all_ancestries_call_rate_pass` or `all_ancestries_hwe_pass` from `collated_variant_qc.tsv` are excluded. MAF is **not** required for correction — rare variant noise is handled at the association stage. |

**INTENSITY_ONLY markers** are **retained** for correction because their
LRR values are informative for batch-effect removal, even though they lack
genotype clusters.

All thresholds are configurable via `--min-marker-call-rate`,
`--min-var`, `--max-var`, `--no-complexity-filter`, `--variant-qc`, or
the `marker_qc` section of the YAML config.

### Association Stage (GWAS Testing)

The `associate` sub-command applies minimal hard-fail marker exclusion
**before the per-marker regression**.  Upstream variant QC flags are
**not** used to pre-filter LRR markers — they are propagated to the
output TSV for post-hoc filtering.

| Filter | Default | Rationale |
|--------|---------|-----------|
| **INTENSITY_ONLY** | Excluded | Non-polymorphic probes flagged as `INTENSITY_ONLY` in the BCF `INFO` field have no genotype cluster (no GT). Their LRR values are not comparable to genotyped markers and should not be tested for association. |
| **Monomorphic LRR** | Excluded | Markers with zero LRR variance across analysed samples are uninformative and produce degenerate test statistics. |

Each filter can be **disabled individually**:

| CLI Flag | Config Key | Effect |
|----------|------------|--------|
| `--no-exclude-intensity-only` | `association_marker_qc.exclude_intensity_only: false` | Retain INTENSITY_ONLY markers in association |
| `--no-exclude-monomorphic-lrr` | `association_marker_qc.exclude_monomorphic_lrr: false` | Retain zero-variance markers |

> **Upstream variant QC flags are NOT used to pre-filter LRR association
> markers.**  Instead, per-marker QC flags (`all_ancestries_call_rate_pass`,
> `all_ancestries_hwe_pass`, `all_ancestries_maf_pass`) are propagated to
> the output TSV for every tested marker, enabling trivial post-hoc
> filtering without re-running the scan.  Additional provenance columns
> (`intensity_only`, `lrr_monomorphic`) are also included.

This design follows modern GWAS pipeline conventions (SAIGE, REGENIE):
**run association once on the maximally inclusive marker set, annotate
everything, filter the output freely.**  This is especially appropriate
for LRR where genotype-derived QC metrics (MAF, HWE) are orthogonal to
the continuous intensity signal being tested.

The log output includes a per-step summary:

```
INFO: Association marker exclusion: INTENSITY_ONLY: 342 / 96869 excluded
INFO: Association marker exclusion: monomorphic LRR (zero variance): 17 / 96869 excluded
INFO: Association marker exclusion summary: 96510 / 96869 markers pass all filters (359 excluded, 0.4%)
```

### GRM Marker Filtering (LMM Only)

When `--method lmm` is used, genotype markers for the GRM undergo
separate, stricter filtering:

1. MAF ≥ 0.01 (`--min-maf`)
2. Genotype call rate ≥ 0.90 (`--min-gt-call-rate`)
3. Upstream variant QC: call rate + HWE + MAF (`--variant-qc`)
4. LD pruning: window 1 Mb, r² < 0.2 (`--ld-window-bp`, `--ld-r2-thresh`)

### References

- Anderson CA, Pettersson FH, Clarke GM, et al. (2010). Data quality
  control in genetic case-control association studies. *Nature Protocols*,
  5(9), 1564–1573.
- Marees AT, de Kluiver H, Stringer S, et al. (2018). A tutorial on
  conducting genome-wide association studies: Quality control and
  statistical analysis. *International Journal of Methods in Psychiatric
  Research*, 27(2), e1608.

---

## QC Configuration

Sample and marker QC thresholds can be set via a YAML configuration file
passed with `--config` to `correct`. The same config can be passed to
`associate` to provide `upstream_qc.variant_qc_path` for GRM marker
filtering and `association_qc` settings for sample exclusion.
CLI flags take precedence over YAML values, which take precedence over
built-in defaults.

### Default Configuration

```yaml
sample_qc:
  max_lrr_sd: 0.35            # Max LRR SD for HQ classification
  min_call_rate: 0.97          # Min call rate for HQ classification

association_qc:
  honor_precomputed: true      # Honor pre_pca_excluded, excluded_relatedness,
                               #   excluded_het_outlier from sample sheet
  exclude_baf_sd: true         # Exclude samples with baf_sd > max_baf_sd
  max_baf_sd: 0.15             # BAF SD threshold (contamination proxy)
  exclude_sex_discordant: true # Exclude sex_status == "DISCORDANT"
  exclude_extreme_inbreeding: true  # Exclude |inbreeding_F| > threshold
  max_abs_inbreeding_f: 0.15  # Inbreeding F threshold

marker_qc:
  min_call_rate: 0.95          # Min marker call rate
  min_var: 0.001               # Min LRR variance
  max_var: null                # No upper limit

association_marker_qc:
  exclude_intensity_only: true  # Exclude INTENSITY_ONLY probes from association
  exclude_monomorphic_lrr: true # Exclude zero-variance LRR markers

correction:
  k: null                      # Auto via Marchenko-Pastur (PCs removed)
  n_components: null           # Auto = 5% of HQ sample count (PCs computed for auto-k)
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

association_qc:
  max_baf_sd: 0.10               # Stricter contamination threshold
  max_abs_inbreeding_f: 0.10     # Stricter inbreeding threshold

marker_qc:
  min_call_rate: 0.98
  min_var: 0.002
  max_var: 5.0

correction:
  k: 5
  n_components: 50             # Optional; only used when k is null
  backend: rsvd
```

### Permissive Example (Disable Optional Exclusions)

```yaml
association_qc:
  honor_precomputed: false       # Keep related samples and het outliers
  exclude_baf_sd: false          # No BAF SD filter
  exclude_sex_discordant: false  # Keep sex-discordant samples
  exclude_extreme_inbreeding: false  # No inbreeding filter
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

Tab-separated file with a header row. It must contain `sample_id` and at least
one additional numeric column. By default, the first non-`sample_id` column is
used as the phenotype (continuous or binary 0/1), and all other non-`sample_id`
columns are treated as covariates. You can override this with
`--phenotype-col` and `--covariate-cols`. Columns **must** be separated by
literal tab characters (`\t`).

| sample\_id | phenotype | age | sex |
|------------|-----------|-----|-----|
| SAMPLE\_001 | 0.52 | 45 | 1 |
| SAMPLE\_002 | -0.13 | 62 | 0 |
| SAMPLE\_003 | 1.00 | 38 | 1 |

### Compiled Sample Sheet

A tab-separated file from the upstream pipeline containing sample QC metrics
and ancestry PCs (`PC1`–`PC20`). Passed via `--sample-sheet` to the
`associate` command for sample-QC filtering (and sex-chromosome modes).
Association phenotype/covariate values are read from `--phenotype`.
See [docs/upstream_qc_formats.md](docs/upstream_qc_formats.md) for the full
specification.

**Column casing:** The default sample-ID column is `Sample_ID` (matching
the upstream `illumina_idat_processing` convention), but column names are
resolved **case-insensitively** — so `sample_id`, `SAMPLE_ID`, etc. all
work without extra configuration.

---

## Output Formats

### Corrected BCF/VCF (`correct`)

Same structure as the input BCF/VCF with the `FORMAT/LRR` values replaced by
batch-corrected values.  A `batch_lrr_correction` header line records the
parameters used (k, backend, n\_hq\_samples, n\_markers\_used, singular
values, timestamp).

### SVD Text Outputs (`correct`)

`correct` also writes tab-separated SVD summary files. By default, the prefix
is `<OUTPUT>.svd`; override with `--svd-output-prefix`.

1. `<prefix>.sample_pcs.tsv`
   - Header: `SAMPLE`, `PC1`, `PC2`, ..., `PCk`
   - One row per sample in BCF/VCF sample order
   - Values: sample PC values for the correction components (`diag(s) @ Vᵀ`)

2. `<prefix>.singular_values.tsv`
   - Header: `PC`, `singular_value`, `used_for_correction`
   - One row per component (`PC1`..`PCk`)

3. `<prefix>.correction_info.json`
   - JSON sidecar written by `correct`; required by `diagnostic-report`
   - Fields: `k` (int), `n_hq_samples` (int), `n_markers_used` (int),
     `hq_sample_mask` (list of bool, one entry per sample)

4. `<prefix>.sample_metrics.tsv`
   - Per-sample QC metrics written alongside every report
   - Header: `SAMPLE`, `LRR_SD`, `callrate`, `n_markers_used`
   - When post-correction metrics are available: also `LRR_SD_post`,
     `callrate_post`

5. `<prefix>.umap.tsv` *(written when UMAP is computed)*
   - Header: `SAMPLE`, `umap1`, `umap2`
   - Pre-computed 2-D UMAP embedding of samples (from `max(3, k_MP)` PCs)
   - Written during `correct`; loaded by `diagnostic-report` to avoid
     recomputing

6. `<prefix>.sample_summary.tsv` — **Consolidated per-sample table**
   - One row per sample; all key per-sample data in one file
   - Columns (in order):
     - `sample_id` — sample identifier
     - `LRR_SD` — pre-correction LRR standard deviation
     - `callrate` — pre-correction call rate
     - `n_markers_used` — number of autosomal markers used for metrics
     - `LRR_SD_post`, `callrate_post` — post-correction equivalents (if available)
     - `hq` — 1 = HQ, 0 = LQ
     - `umap1`, `umap2` — UMAP coordinates (if computed)
     - All additional columns from `--sample-sheet` (and `--illumina-sample-sheet`)
   - Suitable as input to downstream visualisation scripts, R/Pandas analyses,
     or supplementary tables

Optional (`--write-loadings`):

7. `<prefix>.loadings.tsv`
   - Header: `chrom`, `pos`, `variant_id`, `PC1`, `PC2`, ..., `PCk`
   - One row per marker retained for decomposition (`marker_mask=True`)
   - Values: marker loadings (`U`) used during residualisation

### Interactive Diagnostic Report (`correct`)

By default, `correct` generates a self-contained interactive HTML report at
`<OUTPUT>.diagnostic_report.html` and a sample metrics TSV at
`<prefix>.sample_metrics.tsv`.  Disable with `--no-interactive-report`.

The report contains:

- **Scree plot** — eigenvalues with the Marchenko–Pastur (MP) cutoff line
  clearly separating significant PCs from noise.
- **3-D PC scatter** — interactive scatter plot of samples on PCs 1–3
  (all PCs up to and including the MP cutoff are selectable).  Supports
  colour overlays by LRR_SD, call rate, and HQ/LQ status.
- **UMAP projection** — 2-D UMAP computed from `max(3, k_MP)` PCs, with
  the same colour overlay options.
- **Pre vs Post-Correction LRR_SD scatter** — paired scatter plot comparing
  pre- and post-correction LRR_SD for each sample, with a unity line
  (y = x).  Points below the line indicate reduced noise after PC
  correction.  Colour distinguishes HQ (blue) and LQ (red) samples.
- **Bland–Altman plot** — difference (pre − post) vs mean LRR_SD per sample,
  with mean-difference and ±1.96 SD limits-of-agreement lines for QC
  assessment of correction efficacy.

The sample metrics TSV has columns `SAMPLE`, `LRR_SD`, `callrate`,
`n_markers_used`, and — when PC correction is applied — `LRR_SD_post` and
`callrate_post`.  Pre- and post-correction metrics are computed on the same
autosomal marker set (intersected with upstream variant QC when available)
to enable direct comparison.

Interactive report dependencies are now included in the base package install:

```bash
pip install array_lrr_gwas
```

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

When `--variant-qc` is provided, the following **QC provenance columns**
are appended for every tested marker:

| Column | Type | Description |
|--------|------|-------------|
| `all_ancestries_call_rate_pass` | bool/empty | Marker passes cross-ancestry call-rate threshold |
| `all_ancestries_hwe_pass` | bool/empty | Marker passes cross-ancestry HWE threshold |
| `all_ancestries_maf_pass` | bool/empty | Marker passes cross-ancestry MAF threshold |
| `all_ancestries_qc_pass` | bool/empty | Optional upstream composite flag (typically call rate + HWE + MAF all pass) |
| `intensity_only` | bool | Marker is an INTENSITY\_ONLY probe (no GT cluster) |
| `lrr_monomorphic` | bool | Marker has zero LRR variance across analysed samples |

> **No marker pre-filtering at the association stage.** By default, only
> `INTENSITY_ONLY` probes and monomorphic-LRR markers are excluded from
> association testing.  Upstream variant QC flags
> (`all_ancestries_call_rate_pass`, `all_ancestries_hwe_pass`,
> `all_ancestries_maf_pass`, and optional `all_ancestries_qc_pass`) are
> **not** used to pre-filter markers —
> instead, they are propagated to the output TSV as provenance columns.
> This enables trivial post-hoc filtering in R, Python, or any
> downstream tool without re-running the association scan.

Example row (tab-separated, with header for reference):

```
chrom  pos    variant_id      beta    se     stat   p_value  n_samples  method  all_ancestries_call_rate_pass  all_ancestries_hwe_pass  all_ancestries_maf_pass  all_ancestries_qc_pass  intensity_only  lrr_monomorphic
chr1   12345  chr1:12345:A:T  -0.023  0.011  -2.09  0.037    4800       lmm     True                           True                     True                     True                    False           False
```

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
- **Sex-stratified analyses drop constant covariates automatically.**
  In sex-stratified modes (`x_male_only`, `x_female_only`, `y_male_only`)
  all analysed samples are of the same sex, so any sex covariate in the
  phenotype file would be constant (zero variance) and uninformative.
  The pipeline scans covariates for each mode before fitting, automatically
  drops constant columns, and emits a `WARNING` log line listing the
  dropped names.  The `x_with_sex_covariate` mode adds sex explicitly as
  a covariate after this scan, so it is never dropped in that mode.
- **Non-autosomal analyses default to all seven modes.**  When
  `--sample-sheet` is provided with a `predicted_sex` column, all seven
  non-autosomal modes (`x_with_sex_covariate`, `x_male_only`,
  `x_female_only`, `y_male_only`, `mt_with_sex_covariate`,
  `mt_male_only`, `mt_female_only`) run automatically.  Pass
  `--sex-chr-mode` with no arguments to opt out entirely.
- **X-GRM requires genotype dosages.**  The X-chromosome GRM is
  computed from `FORMAT/GT` dosages, not from LRR.  If no chrX genotype
  data is available (e.g. the `--genotype-bcf` lacks chrX markers),
  chrX sex-chromosome modes fall back to OLS regression.
- **PAR exclusion depends on `--build`.**  Without a resolved genome
  build (via `--build` or auto-detection), pseudoautosomal regions
  cannot be excluded from the X-GRM.  The X-GRM will still be computed
  but may include PAR variants that segregate like autosomes, slightly
  biasing X-linked relatedness estimates.
- **Male dosage rescaling heuristic.**  The automatic 0/1 → 0/2
  rescaling of male chrX dosages uses a max-dosage check (≤ 1.05) to
  avoid corrupting data already on the 0/2 scale.  In rare cases where
  imputed male dosages have values near 1.0 (e.g. low-confidence
  imputation), the heuristic skips rescaling and logs a debug message.
  Verify that input genotype dosages use the expected 0/2 coding for
  males.

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
├── genome_build.py      # Build detection + exclusion/PAR regions
├── qc_config.py         # YAML configuration with override hierarchy
├── io_vcf.py            # BCF/VCF I/O for LRR
├── genotypes.py         # Genotype extraction (FORMAT/GT → dosage)
├── grm.py               # Genetic Relationship Matrix (autosomal + X-GRM)
├── variant_qc.py        # Variant QC parsing (autosomal + chrX)
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
- Marees, A. T. et al. *A tutorial on conducting genome-wide association
  studies: Quality control and statistical analysis.* International Journal
  of Methods in Psychiatric Research **27**, e1608 (2018).
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
