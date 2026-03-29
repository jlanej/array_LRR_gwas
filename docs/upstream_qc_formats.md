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

1. **`call_rate ≥ min_call_rate`** (upstream default: 0.97, `array_lrr_gwas` default: 0.97)
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
| `autosomes_only` | Restrict to autosomal chromosomes (excludes X, Y, MT) | True |
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
  n_components: null         # Pilot decomposition size for auto-k (null = 5% of HQ sample count)
  backend: rsvd              # "rsvd" (scikit-learn) or "fbpca" (Facebook PCA)
  no_complexity_filter: false # Skip centromere / segdup exclusion regions

# upstream_qc: Ancestry-informed variant QC (see "Upstream Variant QC Integration").
upstream_qc:
  variant_qc_path: null      # Path to collated_variant_qc.tsv (null = no upstream filter)
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
  n_components: 50         # optional pilot decomposition size used only when k is null
  backend: rsvd
```

CLI flags always take precedence over values in the YAML file.

---

## Upstream Variant QC Integration

`array_lrr_gwas` can consume the ancestry-aware variant QC output
(`collated_variant_qc.tsv`) produced by the upstream
`jlanej/illumina_idat_processing` pipeline.  This section explains the
logic, thresholds, and the order of operations.

### Ancestry-Aware QC Logic

The upstream pipeline computes per-variant QC metrics **within each
ancestry stratum** and collapses the results into three cross-ancestry
boolean flags:

| TSV column | Upstream threshold | Interpretation |
|---|---|---|
| `all_ancestries_call_rate_pass` | call rate ≥ 0.98 per ancestry | Marker has adequate genotyping across **all** ancestries |
| `all_ancestries_hwe_pass` | HWE *p* ≥ 1 × 10⁻⁶ per ancestry | Marker is in Hardy-Weinberg equilibrium in **all** ancestries |
| `all_ancestries_maf_pass` | MAF ≥ 0.01 per ancestry | Marker is polymorphic in **all** ancestries |

A variant **must pass a flag in every ancestry** for that flag to be
`True`.  This conservative intersection ensures that included markers
are well-behaved across the full multi-ancestry cohort.

### Integration Sequence

When a `collated_variant_qc.tsv` file is supplied (via `--variant-qc`
or `upstream_qc.variant_qc_path` in the YAML config), `array_lrr_gwas`
applies the following processing sequence:

1. **Load** — Parse the TSV via
   `variant_qc.read_collated_variant_qc()`.
2. **Build QC mask** — For each marker in the input BCF/VCF, build a
   boolean keep-mask using the flags above.  The required flags differ
   by context:
   - **RSVD batch correction**: call rate + HWE (MAF **not** required).
   - **GRM construction**: call rate + HWE + MAF (all three required).
3. **LD prune (GRM only)** — After the QC mask is applied, GRM markers
   are LD-pruned (default: r² < 0.2, 1 Mb window) to prevent highly
   linked regions from dominating the GRM eigenstructure.
4. **Downstream analysis** — The filtered (and optionally LD-pruned)
   marker set is used for batch correction or GRM computation, and
   the association scan runs on the full set of input markers.

If the TSV file is **not** provided, `array_lrr_gwas` logs a warning
and falls back to an all-pass mask (no upstream filtering).

### YAML Configuration with `variant_qc_path`

Add the `upstream_qc` section to point at the collated TSV:

```yaml
# Full example config showing all sections including upstream QC.
sample_qc:
  max_lrr_sd: 0.35
  min_call_rate: 0.97

marker_qc:
  min_call_rate: 0.95
  min_var: 0.001
  max_var: null

correction:
  k: null
  backend: rsvd
  no_complexity_filter: false

# upstream_qc: Ancestry-informed variant QC from illumina_idat_processing.
upstream_qc:
  variant_qc_path: /path/to/collated_variant_qc.tsv
```

The `--variant-qc` CLI flag overrides `upstream_qc.variant_qc_path`
from the YAML config.

### CLI Interaction: `--variant-qc` and LD Pruning

The `--variant-qc` flag and the LD pruning flags work together in
the `associate` sub-command:

1. Markers are first filtered by `--variant-qc` (call rate + HWE +
   MAF pass required for GRM).
2. The surviving markers are then LD-pruned unless `--no-ld-prune` is
   set.
3. LD pruning uses `--ld-window-bp` (default 1 000 000) and
   `--ld-r2-thresh` (default 0.2).

For the `correct` sub-command, `--variant-qc` applies call rate + HWE
filters (MAF not required) to the RSVD marker selection; LD pruning
is not performed during batch correction.

### Provenance in Association Output

When `--variant-qc` is provided for the `associate` sub-command, the
output TSV includes three additional boolean columns for each marker:

| Column | Description |
|---|---|
| `all_ancestries_call_rate_pass` | Whether the marker passed the cross-ancestry call-rate filter |
| `all_ancestries_hwe_pass` | Whether the marker passed the cross-ancestry HWE filter |
| `all_ancestries_maf_pass` | Whether the marker passed the cross-ancestry MAF filter |

Markers that are **not** present in the upstream QC file receive
empty values in these columns.  This allows downstream users and
auditors to trace exactly which markers were included or excluded
by which filters.

---

## References

- Anderson C.A. et al. *Data quality control in genetic case-control
  association studies.* Nat Protoc 5, 1564–1573 (2010).
- Marees A.T. et al. *A tutorial on conducting genome-wide association
  studies.* Int J Methods Psychiatr Res 27:e1608 (2018).
- Turner S. et al. *Quality Control Procedures for Genome-Wide
  Association Studies.* Curr Protoc Hum Genet, Unit 1.19 (2011).
