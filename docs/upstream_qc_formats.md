# Upstream QC Annotation Formats

This document surveys the quality-control (QC) fields produced by the
upstream [`jlanej/illumina_idat_processing`](https://github.com/jlanej/illumina_idat_processing)
pipeline and describes how `array_lrr_gwas` consumes them.

---

## BCF FORMAT Fields

The upstream pipeline converts Illumina IDAT files to BCF/VCF via
`bcftools +gtc2vcf`.  The resulting per-sample FORMAT fields relevant
to LRR-GWAS are:

| Field | Type  | Description |
|-------|-------|-------------|
| `GT`  | String | Genotype (e.g. `0/1`, `./.` for missing) |
| `LRR` | Float  | Log R Ratio — normalised total signal intensity relative to the expected cluster position.  The primary predictor in LRR-GWAS. |
| `BAF` | Float  | B Allele Frequency — allelic ratio (0, 0.5, or 1 for canonical clusters). Used upstream for contamination metrics and mosaic detection but not directly by `array_lrr_gwas`. |

### BCF INFO Fields

| Field | Type | Description |
|-------|------|-------------|
| `INTENSITY_ONLY` | Flag | Marks probes that report intensity but have no genotype cluster (e.g. non-polymorphic CNV probes). The upstream pipeline excludes these from call-rate and genotype QC computation. |

---

## Sample-Level QC

Computed by `scripts/collect_qc_metrics.sh` → `scripts/compute_sample_qc.py`
in a single streaming pass over autosomal variants.

| Column | Type | Description | Default threshold |
|--------|------|-------------|-------------------|
| `sample_id`  | str   | Sample identifier (derived from GTC filename) | — |
| `call_rate`  | float | Fraction of non-missing genotypes (autosomes only) | ≥ 0.97 |
| `lrr_sd`     | float | Standard deviation of LRR (autosomes only). Primary noise metric. | ≤ 0.35 |
| `lrr_mean`   | float | Mean of LRR (autosomes only) | — |
| `lrr_median` | float | Median of LRR (autosomes only), computed via O(n) partition | — |
| `baf_sd`     | float | Standard deviation of BAF for heterozygous SNPs (contamination proxy) | — |
| `het_rate`   | float | Fraction of heterozygous genotype calls | — |
| `computed_gender` | str | Predicted sex from X-chromosome heterozygosity (`M`/`F`) | — |

### HQ / LQ Sample Classification

The upstream `scripts/filter_qc_samples.py` and `array_lrr_gwas.correction.classify_samples`
both use the same two metrics for high-quality classification:

1. **`call_rate ≥ min_call_rate`** (upstream default: 0.97, `array_lrr_gwas` default: 0.95)
2. **`lrr_sd ≤ max_lrr_sd`** (upstream default: 0.35, `array_lrr_gwas` default: 0.35)

Samples failing either threshold are classified as **low-quality (LQ)**.
LQ samples are still included in the corrected output — their batch PCs
are estimated by projection onto the HQ-derived loadings (see
`correction.extrapolate_pcs`).

---

## Marker-Level QC

Computed by `scripts/compute_variant_qc.sh` (via `plink2 --missing`,
`--hardy`, `--freq`) and collated by `scripts/collate_variant_qc.py`.

| Metric | Type | Description | Default threshold |
|--------|------|-------------|-------------------|
| `call_rate` | float | 1 − missingness rate per variant | ≥ 0.98 (upstream) / ≥ 0.95 (`array_lrr_gwas`) |
| `hwe_p`     | float | Hardy-Weinberg equilibrium p-value (per ancestry) | ≥ 1×10⁻⁶ |
| `maf`       | float | Minor allele frequency | ≥ 0.01 |

For `array_lrr_gwas` batch-effect correction, marker subsetting uses:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_call_rate` | Marker call-rate filter | 0.95 |
| `min_var`       | Minimum per-marker LRR variance (removes uninformative markers) | 0.001 |
| `max_var`       | Maximum per-marker LRR variance (removes artefactual markers) | None |
| Complexity exclusion | Centromere / segmental duplication regions | Build-specific |

---

## Compiled Sample Sheet

Produced by `scripts/compile_sample_sheet.py`, merging sample QC with
ancestry PCA projections and optional peddy metrics.  Consumed by
`array_lrr_gwas.sample_sheet.read_sample_sheet`.

| Column group | Columns | Description |
|--------------|---------|-------------|
| Sample QC | `sample_id`, `call_rate`, `lrr_sd`, `lrr_mean`, `lrr_median`, `baf_sd`, `het_rate`, `computed_gender` | Per-sample metrics from Stage 2 |
| Inbreeding | `inbreeding_F` | Plink2 inbreeding coefficient (optional) |
| Global PCs | `PC1` … `PC20` | Full-cohort ancestry PCA projections |
| Ancestry PCs | `{ANC}_PC1` … `{ANC}_PC20` | Per-ancestry PCA (NaN for non-members) |
| Peddy | `peddy_het_ratio`, `peddy_ancestry-prediction`, … | Peddy QC metrics (optional) |

---

## Data Shapes

| Object | Shape | Description |
|--------|-------|-------------|
| LRR matrix | `(n_variants, n_samples)` | Primary input; `np.nan` for missing |
| HQ sample mask | `(n_samples,)` bool | True for samples passing QC |
| Marker mask | `(n_markers,)` bool | True for markers passing QC |
| Corrected LRR | `(n_variants, n_samples)` | Same shape as input, batch-corrected |
| Sample sheet covariates | `(n_samples, n_pcs + n_extra)` | Aligned to BCF sample order |

---

## YAML Configuration Override

All QC thresholds can be overridden with a YAML configuration file
passed via `--config`.  See `array_lrr_gwas.qc_config` for the schema.

Precedence: **CLI flags > YAML config > built-in defaults**.

### Complete Default Configuration

The following YAML shows every recognised key with its best-practice
default value.  Omitted keys keep these defaults automatically:

```yaml
# sample_qc: Controls HQ / LQ sample classification.
sample_qc:
  max_lrr_sd: 0.35          # Max LRR SD (upstream default, filter_qc_samples.py)
  min_call_rate: 0.97        # Min call rate (upstream default, Anderson et al. 2010)

# marker_qc: Controls marker subsetting for batch-correction SVD.
marker_qc:
  min_call_rate: 0.95        # Min marker call rate
  min_var: 0.001             # Min LRR variance (removes uninformative markers)
  max_var: null              # Max LRR variance (null = no upper limit)

# correction: SVD decomposition parameters.
correction:
  k: null                    # Batch PCs to remove (null = auto via Marchenko-Pastur)
  backend: rsvd              # "rsvd" (scikit-learn) or "fbpca" (Facebook PCA)
  no_complexity_filter: false # Skip centromere / segdup exclusion regions
```

### Example: Stricter Thresholds

```yaml
sample_qc:
  max_lrr_sd: 0.30        # stricter noise threshold
  min_call_rate: 0.98      # stricter call-rate threshold

marker_qc:
  min_call_rate: 0.98
  min_var: 0.002
  max_var: 5.0

correction:
  k: 5                     # fix number of batch components
  backend: rsvd
```

CLI flags always take precedence over values in the YAML file.

---

## References

- Anderson C.A. et al. *Data quality control in genetic case-control
  association studies.* Nat Protoc 5, 1564–1573 (2010).
- Marees A.T. et al. *A tutorial on conducting genome-wide association
  studies.* Int J Methods Psychiatr Res 27:e1608 (2018).
- Turner S. et al. *Quality Control Procedures for Genome-Wide
  Association Studies.* Curr Protoc Hum Genet, Unit 1.19 (2011).
