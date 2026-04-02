# Provenance: NGS-PCA `sample_qc.tsv`

## Summary

`sample_qc.tsv` is a per-sample coverage quality-control table derived from
the **1000 Genomes Project high-coverage whole-genome sequencing**
(1000G-highcov) cohort using the
[NGS-PCA](https://github.com/jlanej/NGS-PCA) pipeline.

The canonical copy is fetched at runtime from the upstream repository:

```
https://raw.githubusercontent.com/jlanej/NGS-PCA/refs/heads/master/example/1000G_highcov/output/qc_output/sample_qc.tsv
```

The file schema described here reflects the **post-[PR #34](https://github.com/jlanej/NGS-PCA/pull/34)** format
(28 columns), in which `SUPERPOPULATION` is correctly derived from
the population code and a new `FAMILY_ROLE` column carries the pedigree
relationship field that was previously misplaced into `SUPERPOPULATION`.

---

## Source dataset

| Property | Value |
|---|---|
| Dataset | 1000 Genomes Project (phase 3) high-coverage WGS resequencing |
| Reference genome | GRCh38 |
| Samples | 3 203 unrelated and family-member samples across 26 populations |
| Study accession | ERP114329 (2 504 samples, NYGC re-sequencing) and ERP120144 (698 samples) |
| Sequencing centre | New York Genome Center (NYGC) |
| Instrument | Illumina NovaSeq 6000 (majority); a minority processed via `ILLUMINA` library prep |
| Public availability | <https://www.internationalgenome.org/> |

---

## Generating pipeline

The file was produced with the **NGS-PCA** software suite:

- **Repository:** <https://github.com/jlanej/NGS-PCA>
- **Example inputs:** `example/1000G_highcov/`
  (<https://github.com/jlanej/NGS-PCA/tree/master/example/1000G_highcov>)

NGS-PCA estimates per-sample coverage metrics from WGS BAM/CRAM files,
computes a mitochondrial DNA copy number (mtDNA-CN) estimate, and performs
principal component analysis of coverage variation for population-stratification
correction.

The key computation for **`MTDNA_CN`** follows a coverage-ratio approach:

```
MTDNA_CN ≈ (mean mitochondrial coverage / mean autosomal coverage) × ploidy
```

where ploidy is 2 for a diploid autosome. This is reflected in the approximate
relationship between the `MITO_COV_RATIO` column and `MTDNA_CN`
(`MTDNA_CN ≈ MITO_COV_RATIO × 2`).

---

## Column descriptions

| Column | Type | Description |
|---|---|---|
| `SAMPLE_ID` | string | 1000G sample identifier (e.g. `HG00096`) |
| `MEAN_AUTOSOMAL_COV` | float | Mean per-base coverage across autosomal chromosomes |
| `X_COV_RATIO` | float | Ratio of chrX coverage to autosomal coverage (used for sex inference) |
| `Y_COV_RATIO` | float | Ratio of chrY coverage to autosomal coverage (used for sex inference) |
| `INFERRED_SEX` | string | Sex inferred from coverage ratios (`M` or `F`) |
| `MITO_COV_RATIO` | float | Ratio of mitochondrial coverage to mean autosomal coverage |
| `MEDIAN_GENOME_COV` | float | Median per-base genome-wide coverage |
| `PCT_GENOME_COV_10X` | float | Percentage of genome covered at ≥ 10× depth |
| `PCT_GENOME_COV_20X` | float | Percentage of genome covered at ≥ 20× depth |
| `SD_COV` | float | Standard deviation of per-bin coverage (all bins) |
| `MAD_COV` | float | Median absolute deviation of per-bin coverage |
| `IQR_COV` | float | Interquartile range of per-bin coverage |
| `MEDIAN_BIN_COV` | float | Median per-bin coverage |
| `HQ_MEDIAN_COV` | float | Median per-bin coverage in high-quality (mappable) bins |
| `HQ_SD_COV` | float | Standard deviation of per-bin coverage in HQ bins |
| `HQ_MAD_COV` | float | Median absolute deviation of per-bin coverage in HQ bins |
| `HQ_IQR_COV` | float | Interquartile range of per-bin coverage in HQ bins |
| `MTDNA_CN` | float | **Estimated mitochondrial DNA copy number per diploid cell** |
| `POPULATION` | string | 1000G three-letter population code (e.g. `GBR`, `FIN`, `CHB`) |
| `SUPERPOPULATION` | string | 1000G super-population code derived from `POPULATION` (`AFR`, `AMR`, `EAS`, `EUR`, `SAS`) |
| `REPORTED_SEX` | string | Reported biological sex from 1000G metadata (`M` / `F`) |
| `FAMILY_ROLE` | string | Declared pedigree role from the IGSR PED file field 8 (`unrel`, `father`, `mother`, `child`, …) |
| `RELATEDNESS` | string | Derived relatedness from parental IDs: `unrelated` (0/0 parental IDs) or `related` (non-zero) |
| `RELEASE_BATCH` | integer | 1000G release batch size (2504 or 698) |
| `CENTER_NAME` | string | Sequencing centre (`NYGC`) |
| `STUDY_ID` | string | ENA/ERP study accession |
| `INSTRUMENT_MODEL` | string | Sequencer model |
| `LIBRARY_NAME` | string | Sample identifier used as library name |

---

## Intended use in this repository

This file serves as the phenotype source for:

1. **Phenotype extraction** — `MTDNA_CN` is extracted by
   `scripts/make_mtdna_cn_phenotype.py` and formatted as an
   `array-lrr-gwas associate --phenotype` input.
2. **Covariate construction** — coverage and population covariates
   (`MEAN_AUTOSOMAL_COV`, `SUPERPOPULATION`, `INFERRED_SEX`, etc.) are added
   separately via `--sample-sheet` when running `array-lrr-gwas associate`.

The script `scripts/make_mtdna_cn_phenotype.py` fetches `sample_qc.tsv`
directly from the upstream URL at runtime and writes the two-column
(`sample_id` / `phenotype`) TSV format consumed by the
`array-lrr-gwas associate` command.

---

## Citation

If you use data derived from this file in published work, please cite:

> Byrska-Bishop M, Evani US, Zhao X, et al.
> **High-coverage whole-genome sequencing of the expanded 1000 Genomes Project
> cohort including 602 trios.**
> *Cell*, 185(18):3426–3440.e19, 2022.
> <https://doi.org/10.1016/j.cell.2022.08.004>

and, for the NGS-PCA processing:

> Lane JM, et al. **Biological and environmental predictors of human
> mitochondrial DNA copy number.** *PLoS Genetics*, 2022 (or current
> citation for the NGS-PCA repository:
> <https://github.com/jlanej/NGS-PCA>).

