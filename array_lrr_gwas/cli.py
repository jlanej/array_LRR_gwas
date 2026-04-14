"""Command-line interface for LRR batch-effect correction, association, and segmentation.

Usage
-----
::

    array-lrr-gwas correct input.bcf -o corrected.bcf [--build GRCh38] [--k 5]
    array-lrr-gwas associate input.bcf --phenotype pheno.tsv -o results.tsv
    array-lrr-gwas associate input.bcf --phenotype pheno.tsv \\
        --sample-sheet compiled_sample_sheet.tsv -o results.tsv
    array-lrr-gwas segment results.tsv -o regions.bed [--strategy hmm]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _variant_id(v: dict) -> str:
    """Build a canonical variant ID string from a variant metadata dict."""
    vid = v.get("id")
    if vid is not None and vid != ".":
        return vid
    alts = v.get("alts") or ()
    return f"{v['chrom']}:{v['pos']}:{v.get('ref', '')}:{':'.join(alts)}"


def _fmt_float(val: float) -> str:
    """Format floats compactly for TSV outputs."""
    return f"{float(val):.10g}"


def _write_svd_text_outputs(
    prefix: Path,
    *,
    info: dict,
    samples: list[str],
    variants: list[dict],
    include_loadings: bool = False,
) -> dict[str, Path]:
    """Write SVD metadata text outputs for the ``correct`` command."""
    k = int(info["k"])
    n_computed = int(info.get("n_components_computed", k))
    singular_values = np.asarray(info["singular_values"], dtype=float)
    sample_scores = np.asarray(info["sample_scores"], dtype=float)
    pc_scores = singular_values[:, np.newaxis] * sample_scores

    pcs_path = Path(f"{prefix}.sample_pcs.tsv")
    with pcs_path.open("w", encoding="utf-8") as fh:
        header = ["SAMPLE"] + [f"PC{i + 1}" for i in range(n_computed)]
        fh.write("\t".join(header) + "\n")
        for sample_idx, sample_id in enumerate(samples):
            pcs = [_fmt_float(pc_scores[pc_idx, sample_idx]) for pc_idx in range(n_computed)]
            fh.write("\t".join([sample_id] + pcs) + "\n")

    sv_path = Path(f"{prefix}.singular_values.tsv")
    with sv_path.open("w", encoding="utf-8") as fh:
        fh.write("PC\tsingular_value\tused_for_correction\n")
        for pc_idx, sval in enumerate(singular_values, start=1):
            used = "yes" if pc_idx <= k else "no"
            fh.write(f"PC{pc_idx}\t{_fmt_float(sval)}\t{used}\n")

    out_paths: dict[str, Path] = {
        "sample_pcs": pcs_path,
        "singular_values": sv_path,
    }

    if include_loadings:
        marker_mask = np.asarray(info["marker_mask"], dtype=bool)
        marker_loadings = np.asarray(info["marker_loadings"], dtype=float)
        kept_idx = np.flatnonzero(marker_mask)
        loadings_path = Path(f"{prefix}.loadings.tsv")
        with loadings_path.open("w", encoding="utf-8") as fh:
            header = ["chrom", "pos", "variant_id"] + [f"PC{i + 1}" for i in range(n_computed)]
            fh.write("\t".join(header) + "\n")
            for row_idx, var_idx in enumerate(kept_idx):
                var = variants[var_idx]
                pcs = [_fmt_float(marker_loadings[row_idx, pc_idx]) for pc_idx in range(n_computed)]
                row = [str(var["chrom"]), str(var["pos"]), _variant_id(var)] + pcs
                fh.write("\t".join(row) + "\n")
        out_paths["loadings"] = loadings_path

    return out_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="array-lrr-gwas",
        description="Batch-effect correction and LMM GWAS for array-based LRR values.",
    )
    sub = parser.add_subparsers(dest="command")

    correct = sub.add_parser(
        "correct",
        help="Apply LRR batch-effect correction to a BCF/VCF file.",
    )
    correct.add_argument(
        "input",
        type=Path,
        help="Input BCF or VCF file with FORMAT/LRR field.",
    )
    correct.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output BCF (.bcf) or VCF (.vcf) file path.",
    )
    correct.add_argument(
        "--build",
        type=str,
        default=None,
        help=(
            "Reference genome build (GRCh37, GRCh38, T2T-CHM13, hg19, hg38, "
            "hs1). Auto-detected from the input file when possible."
        ),
    )
    correct.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Number of batch-effect components to remove from LRR. "
            "Auto-selected via Marchenko–Pastur heuristic if omitted."
        ),
    )
    correct.add_argument(
        "--n-components",
        type=int,
        default=None,
        help=(
            "Number of components to compute in the pilot truncated "
            "decomposition used for auto-selection of --k. "
            "Default: 5%% of HQ sample size. Ignored when --k is provided."
        ),
    )
    correct.add_argument(
        "--no-complexity-filter",
        action="store_true",
        help="Disable the default genomic-complexity region exclusion.",
    )
    correct.add_argument(
        "--max-lrr-sd",
        type=float,
        default=0.35,
        help=(
            "Max per-sample LRR SD for HQ classification (default: 0.35). "
            "Matches upstream illumina_idat_processing best practice. "
            "Samples above this are classified as low-quality (LQ)."
        ),
    )
    correct.add_argument(
        "--min-sample-call-rate",
        type=float,
        default=0.97,
        help=(
            "Min per-sample call rate for HQ classification (default: 0.97). "
            "Matches upstream illumina_idat_processing best practice "
            "(Anderson et al. 2010 recommend >= 0.95-0.98)."
        ),
    )
    correct.add_argument(
        "--min-marker-call-rate",
        type=float,
        default=0.95,
        help=(
            "Min per-marker call rate for batch-correction subsetting "
            "(default: 0.95). A moderately inclusive threshold retains more "
            "markers for the SVD decomposition."
        ),
    )
    correct.add_argument(
        "--min-var",
        type=float,
        default=0.001,
        help=(
            "Min per-marker LRR variance (default: 0.001). "
            "Removes near-constant (uninformative) markers from the "
            "decomposition."
        ),
    )
    correct.add_argument(
        "--max-var",
        type=float,
        default=None,
        help=(
            "Max per-marker LRR variance (default: no upper limit). "
            "Set to exclude artefactual high-variance outlier markers."
        ),
    )
    correct.add_argument(
        "--backend",
        type=str,
        default="rsvd",
        choices=["rsvd", "fbpca"],
        help=(
            "Decomposition backend (default: rsvd). "
            "'rsvd' uses scikit-learn randomised SVD; "
            "'fbpca' uses Facebook PCA (requires fbpca package)."
        ),
    )
    correct.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    correct.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML configuration file for QC thresholds and correction "
            "parameters. CLI flags override values from the config file. "
            "See docs/upstream_qc_formats.md for field descriptions."
        ),
    )
    correct.add_argument(
        "--variant-qc",
        type=Path,
        default=None,
        help=(
            "Path to collated_variant_qc.tsv from upstream pipeline. "
            "When provided, markers failing cross-ancestry call rate or "
            "HWE are excluded from the RSVD decomposition (MAF is not "
            "required for batch correction). Sequence: load TSV → build "
            "QC mask (call rate + HWE) → apply to marker subsetting. "
            "Also configurable via upstream_qc.variant_qc_path in the "
            "YAML config. If neither is set, no upstream filtering is "
            "applied and a warning is logged."
        ),
    )
    correct.add_argument(
        "--svd-output-prefix",
        type=Path,
        default=None,
        help=(
            "Prefix for SVD text outputs written by `correct`. "
            "Default: <output>.svd, producing "
            "<prefix>.sample_pcs.tsv and <prefix>.singular_values.tsv."
        ),
    )
    correct.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help=(
            "Directory for structured audit trail files.  When provided, "
            "a detailed per-stage TSV and JSON summary of included/excluded "
            "markers and samples (with reasons) are written to this "
            "directory.  Enables full provenance tracking."
        ),
    )
    correct.add_argument(
        "--write-loadings",
        action="store_true",
        help=(
            "Also write marker loadings to <prefix>.loadings.tsv "
            "(can be large for dense arrays)."
        ),
    )
    correct.add_argument(
        "--no-interactive-report",
        action="store_true",
        help=(
            "Skip generation of the interactive HTML diagnostic report. "
            "By default an HTML report with scree plot, 3-D PC scatter, "
            "and UMAP projection is written next to the output BCF."
        ),
    )
    correct.add_argument(
        "--sample-sheet",
        type=Path,
        default=None,
        dest="correct_sample_sheet",
        help=(
            "Optional path to compiled_sample_sheet.tsv.  When provided, "
            "all columns are embedded in the diagnostic HTML report as "
            "additional colour-overlay options for the PC scatter and UMAP "
            "plots (e.g. call_rate, lrr_sd, sex_status, ancestry flags)."
        ),
    )
    correct.add_argument(
        "--illumina-sample-sheet",
        type=Path,
        default=None,
        dest="correct_illumina_sample_sheet",
        help=(
            "Optional path to an Illumina-format SampleSheet.csv.  "
            "Illumina sample sheets use a comma-separated format with "
            "section headers ([Header], [Manifests], [Data]).  When "
            "provided, columns from the [Data] section (e.g. Gender, "
            "Sample_Plate, Sample_Group, CallRate) are added to the "
            "diagnostic HTML report as colour-overlay options alongside "
            "any columns from --sample-sheet."
        ),
    )
    correct.add_argument(
        "--illumina-sample-id-col",
        type=str,
        default="Sample_Group",
        dest="correct_illumina_sample_id_col",
        help=(
            "Column in the Illumina SampleSheet.csv [Data] section whose values "
            "match the sample IDs used in the rest of the pipeline (e.g. NA19152).  "
            "Defaults to 'Sample_Group' because Illumina sheets typically store the "
            "short population/sample identifier in that column while 'Sample_ID' "
            "holds a longer Illumina-internal barcode string.  If the specified "
            "column is absent the first column of the [Data] section is used."
        ),
    )
    correct.add_argument(
        "--max-ram-gb",
        type=float,
        default=None,
        dest="max_ram_gb",
        help=(
            "Maximum RAM in GB available for the RSVD decomposition step "
            "(default: no limit). When both --max-ram-gb and --variant-qc "
            "are provided, markers are pre-selected before loading: every "
            "Nth QC-passing marker is chosen so only the budgeted subset is "
            "read from the BCF, and the remaining markers are corrected via "
            "streaming QR regression that never loads the full matrix. "
            "When --variant-qc is absent, the full matrix is loaded and "
            "every-Nth subsampling is applied in memory. "
            "A value of 0 disables subsampling entirely."
        ),
    )

    # ---- associate sub-command ----
    assoc = sub.add_parser(
        "associate",
        help="Run LRR-based GWAS association scan (LMM, OLS, or logistic).",
    )
    assoc.add_argument(
        "input",
        type=Path,
        help="Input BCF or VCF file with FORMAT/LRR field.",
    )
    assoc.add_argument(
        "--phenotype",
        type=Path,
        required=True,
        help=(
            "Tab-separated phenotype file. Must contain a header row with "
            "at least 'sample_id' and 'phenotype' columns.  Additional "
            "columns are treated as covariates."
        ),
    )
    assoc.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output TSV file for association results.",
    )
    assoc.add_argument(
        "--method",
        type=str,
        default="lmm",
        choices=["lmm", "ols", "logistic"],
        help="Association method (default: lmm).",
    )
    assoc.add_argument(
        "--sample-sheet",
        type=Path,
        default=None,
        help=(
            "Path to compiled_sample_sheet.tsv from illumina_idat_processing. "
            "Global ancestry PCs and covariates are extracted automatically."
        ),
    )
    assoc.add_argument(
        "--hq-samples",
        type=Path,
        default=None,
        help=(
            "Optional path to HQ sample list (one sample ID per line, or "
            "tab-separated with sample ID in column 1). When provided, "
            "association uses only the intersection of non-missing phenotype "
            "samples and this HQ list (LQ samples are dropped). "
            "When omitted but --sample-sheet is provided, HQ samples are "
            "derived from the sample sheet using --max-lrr-sd and "
            "--min-sample-call-rate thresholds (same criteria as the "
            "correct sub-command)."
        ),
    )
    assoc.add_argument(
        "--max-lrr-sd",
        type=float,
        default=None,
        help=(
            "Max per-sample LRR SD for HQ classification (default: 0.35). "
            "Used when --hq-samples is omitted and --sample-sheet is "
            "provided, to derive HQ samples from the sample sheet. "
            "Also configurable via sample_qc.max_lrr_sd in the YAML config."
        ),
    )
    assoc.add_argument(
        "--min-sample-call-rate",
        type=float,
        default=None,
        help=(
            "Min per-sample call rate for HQ classification (default: 0.97). "
            "Used when --hq-samples is omitted and --sample-sheet is "
            "provided, to derive HQ samples from the sample sheet. "
            "Also configurable via sample_qc.min_call_rate in the YAML config."
        ),
    )
    assoc.add_argument(
        "--n-pcs",
        type=int,
        default=20,
        help="Number of global ancestry PCs to use from sample sheet (default: 20).",
    )

    # ---- Association sample-exclusion flags ----
    # These flags control additional exclusion criteria applied when
    # --sample-sheet is provided.  Each is ON by default following GWAS
    # best-practice (Anderson et al. 2010; Marees et al. 2018; UK Biobank
    # and TOPMed SOPs).  All can be disabled individually via the flags
    # below or via the association_qc section in the YAML --config.
    assoc.add_argument(
        "--no-honor-precomputed",
        action="store_true",
        default=False,
        help=(
            "Disable honoring pre-computed exclusion columns "
            "(pre_pca_excluded, excluded_relatedness, excluded_het_outlier) "
            "from the sample sheet.  By default these upstream flags are "
            "respected: related samples (typically up to 2nd degree) are "
            "removed to reduce bias in effect estimates even though the "
            "GRM/LMM corrects for moderate kinship, and het-outliers are "
            "removed as potential contamination or sample errors."
        ),
    )
    assoc.add_argument(
        "--no-exclude-baf-sd",
        action="store_true",
        default=False,
        help=(
            "Disable BAF SD exclusion.  By default, samples with "
            "baf_sd > --max-baf-sd (0.15) are excluded as a proxy for "
            "DNA contamination (Marees et al. 2018).  Requires 'baf_sd' "
            "column in the sample sheet; silently skipped if absent."
        ),
    )
    assoc.add_argument(
        "--max-baf-sd",
        type=float,
        default=None,
        help=(
            "BAF SD threshold for sample exclusion (default: 0.15). "
            "Samples with baf_sd above this value are excluded. "
            "Higher BAF SD suggests potential DNA contamination."
        ),
    )
    assoc.add_argument(
        "--no-exclude-sex-discordant",
        action="store_true",
        default=False,
        help=(
            "Disable sex-discordance exclusion.  By default, samples with "
            "sex_status == 'DISCORDANT' in the sample sheet are excluded "
            "as potential sample swaps (Anderson et al. 2010 Nat Protoc)."
        ),
    )
    assoc.add_argument(
        "--no-exclude-extreme-inbreeding",
        action="store_true",
        default=False,
        help=(
            "Disable extreme inbreeding coefficient exclusion.  By default, "
            "samples with |inbreeding_F| > --max-abs-inbreeding-f (0.15) "
            "are excluded as a safety net for extreme population structure "
            "or sample quality issues (Anderson et al. 2010)."
        ),
    )
    assoc.add_argument(
        "--max-abs-inbreeding-f",
        type=float,
        default=None,
        help=(
            "Absolute inbreeding coefficient threshold (default: 0.15). "
            "Samples with |F| exceeding this are excluded.  "
            "Higher values of |F| may indicate extreme population "
            "stratification (F > 0) or contamination (F < 0)."
        ),
    )
    assoc.add_argument(
        "--genotype-bcf",
        type=Path,
        default=None,
        help=(
            "BCF/VCF with FORMAT/GT for GRM computation.  If omitted the "
            "input file is used (requires GT field)."
        ),
    )
    assoc.add_argument(
        "--min-maf",
        type=float,
        default=0.01,
        help=(
            "Minimum MAF for GRM genotype filtering (default: 0.01). "
            "Standard GWAS QC threshold (Marees et al. 2018)."
        ),
    )
    assoc.add_argument(
        "--min-gt-call-rate",
        type=float,
        default=0.90,
        help=(
            "Minimum genotype call rate for GRM filtering (default: 0.90). "
            "A lenient threshold for GRM computation; variant-level QC "
            "should be applied upstream."
        ),
    )
    assoc.add_argument(
        "--no-ld-prune",
        action="store_true",
        default=False,
        help=(
            "Disable LD pruning of GRM markers.  By default, markers "
            "are LD-pruned (after any --variant-qc filtering) so that "
            "highly linked regions do not disproportionately dominate "
            "the GRM eigenstructure.  When --variant-qc is also set, "
            "the QC mask is still applied even if LD pruning is disabled."
        ),
    )
    assoc.add_argument(
        "--ld-window-bp",
        type=int,
        default=1_000_000,
        help=(
            "LD-pruning window size in base pairs (default: 1000000). "
            "Variant pairs within this distance are evaluated for LD."
        ),
    )
    assoc.add_argument(
        "--ld-r2-thresh",
        type=float,
        default=0.2,
        help=(
            "r² threshold for LD pruning (default: 0.2). "
            "Variant pairs with r² above this are pruned."
        ),
    )
    assoc.add_argument(
        "--ld-backend",
        type=str,
        default="plink2",
        choices=["numpy", "plink2"],
        help=(
            "Backend for LD pruning (default: plink2). "
            "'plink2' shells out to plink2 --indep-pairwise for "
            "faster pruning on large datasets; if plink2 is not on PATH "
            "the pipeline will exit with an error.  Use '--ld-backend numpy' "
            "to explicitly select the pure-Python fallback."
        ),
    )
    assoc.add_argument(
        "--variant-qc",
        type=Path,
        default=None,
        help=(
            "Path to collated_variant_qc.tsv from upstream pipeline. "
            "When provided, genotype markers failing cross-ancestry "
            "call rate, HWE, or MAF are excluded before LD pruning "
            "for GRM computation. Sequence: load TSV → QC mask "
            "(call rate + HWE + MAF) → LD prune (unless --no-ld-prune) "
            "→ GRM. Also configurable via upstream_qc.variant_qc_path "
            "in the YAML config. If neither is set, no upstream "
            "filtering is applied and a warning is logged. "
            "Per-marker QC flags (call_rate_pass, hwe_pass, maf_pass) "
            "are propagated to the output TSV for every tested marker, "
            "enabling trivial post-hoc filtering without re-running. "
            "LRR markers are NOT pre-filtered by these flags — only "
            "INTENSITY_ONLY and monomorphic-LRR exclusions are applied "
            "before association testing."
        ),
    )

    # ---- Association marker-exclusion flags ----
    # These flags control variant-level filtering applied to LRR markers
    # before the association test.  INTENSITY_ONLY and monomorphic LRR are
    # the only hard-fail pre-filters.  Upstream variant QC flags are
    # propagated to the output TSV for post-hoc filtering — they are NOT
    # used to pre-filter LRR markers.
    assoc.add_argument(
        "--no-exclude-intensity-only",
        action="store_true",
        default=False,
        help=(
            "Disable exclusion of INTENSITY_ONLY markers from association "
            "testing.  By default, markers flagged as INTENSITY_ONLY in "
            "the BCF INFO field are excluded from the association scan "
            "because they report intensity but have no genotype cluster "
            "(no GT field), making their LRR unreliable for GWAS.  They "
            "are retained during batch-effect correction (the correction "
            "stage only uses LRR values).  Also configurable via "
            "association_marker_qc.exclude_intensity_only in the YAML config."
        ),
    )
    assoc.add_argument(
        "--no-exclude-monomorphic-lrr",
        action="store_true",
        default=False,
        help=(
            "Disable exclusion of markers with zero LRR variance across "
            "analysed samples.  Such markers are uninformative (constant "
            "LRR across all samples) and produce degenerate test statistics.  "
            "Also configurable via "
            "association_marker_qc.exclude_monomorphic_lrr in the YAML config."
        ),
    )
    assoc.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML configuration file. For association, this is used "
            "to read upstream_qc.variant_qc_path when --variant-qc is "
            "not set, and to read sample_qc.max_lrr_sd / "
            "sample_qc.min_call_rate when deriving HQ samples from "
            "the sample sheet.  See docs/upstream_qc_formats.md for the "
            "full configuration schema and examples."
        ),
    )
    assoc.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help=(
            "Directory for structured audit trail files.  When provided, "
            "a detailed per-stage TSV and JSON summary of included/excluded "
            "markers and samples (with reasons) are written to this "
            "directory.  Enables full provenance tracking."
        ),
    )
    assoc.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    # ---- segment sub-command ----
    seg = sub.add_parser(
        "segment",
        help="Segment association results into CNV-associated intervals.",
    )
    seg.add_argument(
        "input",
        type=Path,
        help="Input TSV file from the associate sub-command.",
    )
    seg.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output BED file for segmented intervals.",
    )
    seg.add_argument(
        "--strategy",
        type=str,
        default="hmm",
        choices=["hmm", "threshold"],
        help=(
            "Segmentation strategy (default: hmm).  'hmm' uses a two-state "
            "Hidden Markov Model; 'threshold' uses p-value thresholding with "
            "distance-based merging."
        ),
    )
    seg.add_argument(
        "--p-threshold",
        type=float,
        default=5e-8,
        help=(
            "(threshold strategy) Significance threshold for flagging "
            "markers (default: 5e-8)."
        ),
    )
    seg.add_argument(
        "--max-gap",
        type=int,
        default=1_000_000,
        help=(
            "(threshold strategy) Maximum gap in bp between markers to "
            "merge into one segment (default: 1000000)."
        ),
    )
    seg.add_argument(
        "--min-markers",
        type=int,
        default=1,
        help="Minimum number of markers to keep a segment (default: 1).",
    )
    seg.add_argument(
        "--null-rate",
        type=float,
        default=None,
        help=(
            "(HMM) Exponential rate for null-state emission on -log10(p) "
            "(default: ln(10) ≈ 2.303)."
        ),
    )
    seg.add_argument(
        "--signal-rate",
        type=float,
        default=None,
        help=(
            "(HMM) Exponential rate for associated-state emission on "
            "-log10(p).  Smaller values expect stronger signals "
            "(default: 0.1)."
        ),
    )
    seg.add_argument(
        "--prior-assoc",
        type=float,
        default=None,
        help=(
            "(HMM) Prior probability a marker is in the associated "
            "state (default: 0.001)."
        ),
    )
    seg.add_argument(
        "--transition-prob",
        type=float,
        default=None,
        help=(
            "(HMM) Per-marker state-transition probability "
            "(default: 1e-4)."
        ),
    )
    seg.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    # Suppress verbose debug output from third-party libraries (e.g. numba SSA
    # pass messages, umap) that flood the log when --verbose is active.
    for _noisy_lib in ("numba", "umap"):
        logging.getLogger(_noisy_lib).setLevel(logging.WARNING)

    if args.command == "correct":
        return _run_correct(args)

    if args.command == "associate":
        return _run_associate(args)

    if args.command == "segment":
        return _run_segment(args)

    parser.print_help()
    return 1


def _run_correct(args: argparse.Namespace) -> int:
    """Execute the ``correct`` sub-command."""
    from array_lrr_gwas.audit import AuditLogger
    from array_lrr_gwas.correction import correct_lrr
    from array_lrr_gwas.genome_build import detect_build, get_exclusion_regions
    from array_lrr_gwas.io_vcf import (
        read_bcf_sample_ids,
        read_lrr,
        read_lrr_selected,
        stream_correct_write,
        write_corrected,
    )
    from array_lrr_gwas.qc_config import defaults, load_config, apply_to_correct_args
    from array_lrr_gwas.variant_qc import read_collated_variant_qc, variant_qc_mask

    input_path = args.input
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    # Initialize audit logger
    audit = AuditLogger()

    # Load QC config (YAML file + CLI overrides)
    if args.config is not None:
        logger.info("Loading QC config from %s", args.config)
        cfg = load_config(args.config)
    else:
        cfg = defaults()

    # Build CLI overrides dict (only non-default values override the config)
    cli_overrides = {}
    parser_defaults = {
        "max_lrr_sd": 0.35,
        "min_sample_call_rate": 0.97,
        "min_marker_call_rate": 0.95,
        "min_var": 0.001,
        "max_var": None,
        "k": None,
        "n_components": None,
        "backend": "rsvd",
    }
    for key, default_val in parser_defaults.items():
        val = getattr(args, key, None)
        if val != default_val:
            cli_overrides[key] = val

    correct_kwargs = apply_to_correct_args(cfg, cli_overrides)

    # Resolve max_ram_gb: CLI flag > YAML config > None
    max_ram_gb = getattr(args, "max_ram_gb", None)
    if max_ram_gb is None:
        max_ram_gb = cfg.get("correction", {}).get("max_ram_gb")

    # Load upstream variant QC mask for RSVD marker selection
    variant_qc_path = args.variant_qc or cfg.get("upstream_qc", {}).get("variant_qc_path")
    qc_data = None
    if variant_qc_path is not None:
        variant_qc_path = Path(variant_qc_path)
        logger.info("Loading upstream variant QC from %s", variant_qc_path)
        qc_data = read_collated_variant_qc(variant_qc_path)
    else:
        logger.warning(
            "No upstream variant QC file provided (--variant-qc or "
            "upstream_qc.variant_qc_path).  Skipping ancestry-informed "
            "marker filtering for RSVD.  Provide collated_variant_qc.tsv "
            "for best-practice QC."
        )

    # ----------------------------------------------------------------
    # Decide whether to use the memory-efficient streaming path.
    # Streaming mode is enabled when BOTH max_ram_gb is active AND an
    # upstream variant QC file is available (so we can pre-select
    # markers before scanning the BCF).
    # ----------------------------------------------------------------
    use_streaming = (
        max_ram_gb is not None
        and max_ram_gb > 0
        and qc_data is not None
    )

    if use_streaming:
        return _run_correct_streaming(
            args, input_path, audit, cfg, correct_kwargs,
            max_ram_gb, qc_data, variant_qc_path,
        )
    else:
        correct_kwargs["max_ram_gb"] = max_ram_gb
        return _run_correct_full(
            args, input_path, audit, cfg, correct_kwargs,
            max_ram_gb, qc_data,
        )


def _resolve_exclusion_regions(args, cfg, input_path, chromosomes):
    """Shared helper: resolve genome build and complexity-exclusion regions."""
    from array_lrr_gwas.genome_build import detect_build, get_exclusion_regions

    no_complexity = (
        args.no_complexity_filter
        or cfg["correction"].get("no_complexity_filter", False)
    )
    if no_complexity:
        return None
    build = args.build
    if build is None:
        build = detect_build(input_path)
    if build is None:
        logger.error(
            "Could not detect genome build from input file. "
            "Please supply --build (GRCh37, GRCh38, or T2T-CHM13)."
        )
        return "error"
    logger.info("Using genome build: %s", build)
    exclude_regions = get_exclusion_regions(
        build, chromosomes=list(set(chromosomes))
    )
    n_regions = sum(len(v) for v in exclude_regions.values())
    logger.info(
        "Applying %d default exclusion regions (%s)",
        n_regions, build,
    )
    return exclude_regions


def _write_outputs(args, info, samples, variants, lrr, corrected,
                   chromosomes, upstream_qc_mask, audit,
                   post_metrics=None):
    """Shared helper: write SVD text outputs, report, and audit trail."""
    svd_prefix = args.svd_output_prefix or Path(f"{args.output}.svd")
    svd_paths = _write_svd_text_outputs(
        svd_prefix,
        info=info,
        samples=samples,
        variants=variants,
        include_loadings=args.write_loadings,
    )
    logger.info("Wrote sample PCs: %s", svd_paths["sample_pcs"])
    logger.info("Wrote singular values: %s", svd_paths["singular_values"])
    if "loadings" in svd_paths:
        logger.info("Wrote loadings: %s", svd_paths["loadings"])

    # Generate interactive HTML diagnostic report
    if not getattr(args, "no_interactive_report", False):
        from array_lrr_gwas.interactive_report import generate_report

        report_path = Path(f"{args.output}.diagnostic_report.html")
        metrics_tsv_path = Path(f"{svd_prefix}.sample_metrics.tsv")
        try:
            generate_report(
                info=info,
                samples=samples,
                lrr=lrr,
                corrected_lrr=corrected,
                post_metrics=post_metrics,
                chromosomes=chromosomes,
                upstream_qc_mask=upstream_qc_mask,
                output_path=report_path,
                metrics_tsv_path=metrics_tsv_path,
                sample_sheet_path=getattr(args, "correct_sample_sheet", None),
                illumina_sample_sheet_path=getattr(args, "correct_illumina_sample_sheet", None),
                illumina_sample_id_col=getattr(args, "correct_illumina_sample_id_col", "Sample_Group"),
            )
            logger.info("Wrote interactive report: %s", report_path)
            logger.info("Wrote sample metrics TSV: %s", metrics_tsv_path)
        except Exception:
            logger.warning(
                "Failed to generate interactive report. "
                "Install plotly and umap-learn for full support.",
                exc_info=True,
            )

    # Write audit trail if requested
    audit_dir = getattr(args, "audit_dir", None)
    if audit_dir is not None:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit.write_tsv(audit_dir / "correct_audit.tsv")
        audit.write_json(audit_dir / "correct_audit.json")
        audit.write_summary_tsv(audit_dir / "correct_audit_summary.tsv")

        # Always write the RSVD marker audit when audit_dir is set
        rsvd_audit_path = audit_dir / "rsvd_markers_used.tsv"
        marker_mask_arr = np.asarray(info["marker_mask"], dtype=bool)
        kept_idx = np.flatnonzero(marker_mask_arr)
        with rsvd_audit_path.open("w", encoding="utf-8") as fh:
            fh.write("variant_id\tchrom\tpos\n")
            for idx in kept_idx:
                var = variants[idx]
                fh.write(f"{_variant_id(var)}\t{var['chrom']}\t{var['pos']}\n")
        logger.info(
            "Wrote RSVD marker audit (%d markers): %s",
            len(kept_idx), rsvd_audit_path,
        )


def _run_correct_full(args, input_path, audit, cfg, correct_kwargs,
                      max_ram_gb, qc_data):
    """Non-streaming correction path: loads the full LRR matrix."""
    from array_lrr_gwas.correction import correct_lrr
    from array_lrr_gwas.io_vcf import read_lrr, write_corrected
    from array_lrr_gwas.variant_qc import variant_qc_mask

    # Read full input
    logger.info("Reading LRR from %s", input_path)
    lrr, samples, variants = read_lrr(input_path)
    logger.info(
        "Loaded %d variants × %d samples", lrr.shape[0], lrr.shape[1]
    )

    # Extract positions and chromosomes from variants
    positions = np.array([v["pos"] for v in variants], dtype=np.intp)
    chromosomes = np.array([v["chrom"] for v in variants], dtype=str)

    # Determine exclusion regions
    exclude_regions = _resolve_exclusion_regions(args, cfg, input_path, chromosomes)
    if exclude_regions == "error":
        return 1

    # Build upstream variant QC mask
    upstream_qc_mask = None
    if qc_data is not None:
        variant_ids = [_variant_id(v) for v in variants]
        upstream_qc_mask = variant_qc_mask(
            variant_ids, qc_data,
            require_call_rate=True, require_hwe=True, require_maf=False,
            audit=audit, audit_stage="correction_variant_qc",
        )
        n_pass = int(upstream_qc_mask.sum())
        logger.info(
            "Upstream variant QC (RSVD): %d / %d markers pass "
            "(call rate + HWE; MAF not required)",
            n_pass, len(upstream_qc_mask),
        )

    # Upfront RAM estimation
    if max_ram_gb is not None and max_ram_gb > 0:
        from array_lrr_gwas.decomposition import estimate_rsvd_marker_budget

        n_samples_est = len(samples)
        if upstream_qc_mask is not None:
            n_variants_est = int(upstream_qc_mask.sum())
        else:
            n_variants_est = lrr.shape[0]
        max_ram_bytes = int(max_ram_gb * 1024**3)
        estimated_k = correct_kwargs.get("k") or max(1, int(np.ceil(0.05 * n_samples_est)))
        budget_est = estimate_rsvd_marker_budget(
            n_samples_est, estimated_k, max_ram_bytes=max_ram_bytes
        )
        est_ram_gb = (
            2.5 * n_variants_est * n_samples_est * 8 / 1024**3
        )
        logger.info(
            "RAM estimate: ~%.1f GB for %d variants × %d samples "
            "(budget: %.1f GB, ~%d marker limit). %s",
            est_ram_gb, n_variants_est, n_samples_est,
            max_ram_gb, budget_est,
            "Subsampling likely needed."
            if n_variants_est > budget_est
            else "Should fit within budget.",
        )

    # Run correction
    logger.info(
        "Running batch-effect correction (k=%s, n_components=%s)",
        correct_kwargs.get("k") or "auto",
        correct_kwargs.get("n_components") or "auto(5% of HQ samples)",
    )
    variant_ids = [_variant_id(v) for v in variants]
    corrected, info = correct_lrr(
        lrr,
        positions=positions,
        chromosomes=chromosomes,
        exclude_regions=exclude_regions,
        upstream_qc_mask=upstream_qc_mask,
        audit=audit,
        variant_ids=variant_ids,
        sample_ids=samples,
        **correct_kwargs,
    )
    logger.info(
        "Correction complete: k=%d, %d HQ samples, %d markers used",
        info["k"],
        info["n_hq_samples"],
        info["n_markers_used"],
    )

    # Write output
    logger.info("Writing corrected LRR to %s", args.output)
    write_corrected(
        args.output,
        corrected,
        samples,
        variants,
        info,
        path_template=input_path,
    )

    _write_outputs(args, info, samples, variants, lrr, corrected,
                   chromosomes, upstream_qc_mask, audit)

    logger.info("Done.")
    return 0


def _run_correct_streaming(args, input_path, audit, cfg, correct_kwargs,
                           max_ram_gb, qc_data, variant_qc_path):
    """Memory-efficient streaming correction path.

    Pre-selects markers from the variant QC file, loads only the selected
    subset for SVD, then streams through the full BCF/VCF for QR regression
    and writes corrected output without ever holding the full matrix in RAM.
    """
    from array_lrr_gwas.correction import correct_lrr
    from array_lrr_gwas.decomposition import estimate_rsvd_marker_budget
    from array_lrr_gwas.io_vcf import (
        read_bcf_sample_ids,
        read_lrr_selected,
        stream_correct_write,
    )
    from array_lrr_gwas.subsetting import select_every_nth
    from array_lrr_gwas.variant_qc import variant_qc_mask

    logger.info(
        "Using memory-efficient streaming path (max_ram_gb=%.1f, "
        "variant QC from %s)",
        max_ram_gb, variant_qc_path,
    )

    # 1. Get sample count from BCF header (no LRR loading)
    samples = read_bcf_sample_ids(input_path)
    n_samples = len(samples)
    logger.info("BCF contains %d samples", n_samples)

    # 2. Build list of QC-passing variant IDs (in QC file order)
    passing_ids = [
        vid for vid, rec in qc_data.items()
        if rec.call_rate_pass and rec.hwe_pass
    ]
    logger.info(
        "Upstream variant QC: %d / %d variants pass (call rate + HWE)",
        len(passing_ids), len(qc_data),
    )

    if not passing_ids:
        logger.error("No variants pass upstream QC — cannot proceed.")
        return 1

    # 3. Estimate marker budget and select every-Nth
    estimated_k = correct_kwargs.get("k") or max(1, int(np.ceil(0.05 * n_samples)))
    max_ram_bytes = int(max_ram_gb * 1024**3)
    budget = estimate_rsvd_marker_budget(
        n_samples, estimated_k, max_ram_bytes=max_ram_bytes,
    )
    n_passing = len(passing_ids)

    if n_passing > budget:
        selected_ids = select_every_nth(passing_ids, budget)
        step = max(1, n_passing // budget)
        logger.info(
            "Pre-selected %d / %d QC-passing markers (every-%d, "
            "budget=%d for %.1f GB limit)",
            len(selected_ids), n_passing, step, budget, max_ram_gb,
        )
    else:
        selected_ids = passing_ids
        logger.info(
            "All %d QC-passing markers fit within %.1f GB budget (%d limit); "
            "no subsampling needed.",
            n_passing, max_ram_gb, budget,
        )

    # 4. Load only selected markers from BCF
    logger.info("Scanning BCF to load %d selected markers", len(selected_ids))
    lrr_subset, samples, variants_subset = read_lrr_selected(
        input_path, set(selected_ids),
    )
    logger.info(
        "Loaded subset: %d variants × %d samples (%.2f GB)",
        lrr_subset.shape[0], lrr_subset.shape[1],
        lrr_subset.nbytes / 1024**3,
    )

    if lrr_subset.shape[0] == 0:
        logger.error(
            "No selected markers found in BCF — check that variant IDs in "
            "the QC file match the BCF.  Selected %d IDs but none matched.",
            len(selected_ids),
        )
        return 1

    # 5. Extract positions and chromosomes from the subset
    positions_sub = np.array([v["pos"] for v in variants_subset], dtype=np.intp)
    chromosomes_sub = np.array([v["chrom"] for v in variants_subset], dtype=str)

    # Determine exclusion regions (using subset chromosomes)
    exclude_regions = _resolve_exclusion_regions(
        args, cfg, input_path, chromosomes_sub,
    )
    if exclude_regions == "error":
        return 1

    # 6. Build upstream QC mask for the subset (all True, since pre-selected)
    variant_ids_sub = [_variant_id(v) for v in variants_subset]
    upstream_qc_mask_sub = variant_qc_mask(
        variant_ids_sub, qc_data,
        require_call_rate=True, require_hwe=True, require_maf=False,
        audit=audit, audit_stage="correction_variant_qc",
    )

    # 7. Run SVD on the subset to compute PC scores.
    # Use skip_residualize=True so that markers are NOT corrected in
    # memory here — correction will happen exactly once in step 8 when
    # stream_correct_write() processes ALL markers from the original file.
    # max_ram_gb=None since we already pre-selected to fit within budget
    subset_kwargs = {k: v for k, v in correct_kwargs.items() if k != "max_ram_gb"}
    logger.info(
        "Computing PC scores from %d-marker subset (k=%s, n_components=%s)",
        lrr_subset.shape[0],
        subset_kwargs.get("k") or "auto",
        subset_kwargs.get("n_components") or "auto(5% of HQ samples)",
    )
    _, info = correct_lrr(
        lrr_subset,
        positions=positions_sub,
        chromosomes=chromosomes_sub,
        exclude_regions=exclude_regions,
        upstream_qc_mask=upstream_qc_mask_sub,
        audit=audit,
        variant_ids=variant_ids_sub,
        sample_ids=samples,
        max_ram_gb=None,
        skip_residualize=True,
        **subset_kwargs,
    )
    logger.info(
        "SVD complete: k=%d, %d HQ samples, %d markers used for decomposition",
        info["k"], info["n_hq_samples"], info["n_markers_used"],
    )

    # Record budget metadata in info
    info["rsvd_subsampled"] = n_passing > budget
    info["rsvd_marker_budget"] = budget

    # 8. Extract PC scores and stream-correct ALL markers → output.
    # This is the SINGLE correction pass: every marker in the original
    # BCF is PC-corrected exactly once from the uncorrected source.
    # Post-correction diagnostic metrics are accumulated during streaming
    # for the subset markers, avoiding a separate in-memory correction.
    k = info["k"]
    Vt_k = np.asarray(info["sample_scores"])[:k, :]

    # Build the set of diagnostic marker IDs for streaming accumulation.
    # These are the autosomal, QC-passing markers from the subset — the
    # same set that compute_sample_metrics() would use for pre-correction
    # metrics, ensuring identical marker alignment pre/post.
    from array_lrr_gwas.subsetting import autosome_mask as _auto_mask
    _diag_mask = np.ones(len(variants_subset), dtype=bool)
    if chromosomes_sub is not None:
        _diag_mask &= _auto_mask(chromosomes_sub)
    if upstream_qc_mask_sub is not None:
        _diag_mask &= upstream_qc_mask_sub
    diagnostic_ids = {
        _variant_id(variants_subset[i])
        for i in range(len(variants_subset))
        if _diag_mask[i]
    }
    logger.info(
        "Tracking %d diagnostic markers during streaming for post-correction metrics",
        len(diagnostic_ids),
    )

    logger.info(
        "Streaming single-pass PC correction of ALL markers to %s",
        args.output,
    )
    all_variants, n_skipped, post_metrics = stream_correct_write(
        input_path,
        args.output,
        Vt_k,
        samples,
        info,
        path_template=input_path,
        chunk_size=correct_kwargs.get("chunk_size", 5000),
        min_valid_frac=correct_kwargs.get("min_valid_frac", 0.5),
        diagnostic_marker_ids=diagnostic_ids,
    )
    logger.info(
        "Streaming correction done: %d variants corrected, %d skipped",
        len(all_variants), n_skipped,
    )

    # 9. Write SVD outputs, report, and audit trail.
    # Post-correction metrics were accumulated during streaming;
    # no separate in-memory correction step is needed.
    _write_outputs(
        args, info, samples, variants_subset,
        lrr_subset, None,  # corrected_lrr not needed; post_metrics below
        chromosomes_sub, upstream_qc_mask_sub, audit,
        post_metrics=post_metrics,
    )

    logger.info("Done (streaming mode).")
    return 0


def _run_associate(args: argparse.Namespace) -> int:
    """Execute the ``associate`` sub-command."""
    import csv
    import warnings

    import numpy as np

    from array_lrr_gwas.audit import AuditLogger
    from array_lrr_gwas.association import run_association, run_association_streaming
    from array_lrr_gwas.io_vcf import read_variant_metadata, stream_lrr_chunks
    from array_lrr_gwas.qc_config import defaults, load_config, apply_to_associate_args

    input_path = args.input
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    pheno_path = args.phenotype
    if not pheno_path.exists():
        logger.error("Phenotype file not found: %s", pheno_path)
        return 1

    # Initialize audit logger
    audit = AuditLogger()

    # Optional config (currently used for upstream_qc.variant_qc_path).
    if args.config is not None:
        logger.info("Loading config from %s", args.config)
        cfg = load_config(args.config)
    else:
        cfg = defaults()

    # Read variant metadata and sample IDs without loading LRR values.
    # This is a lightweight scan that keeps memory independent of matrix size.
    logger.info("Scanning variant metadata from %s", input_path)
    samples, variants = read_variant_metadata(input_path)
    n_variants_total = len(variants)
    n_samples_bcf = len(samples)
    logger.info(
        "Found %d variants × %d samples (metadata only, LRR not loaded)",
        n_variants_total, n_samples_bcf,
    )

    # Read phenotype TSV
    logger.info("Reading phenotype from %s", pheno_path)

    def _parse_float(raw: str | None) -> float:
        if raw is None:
            return np.nan
        txt = raw.strip()
        if txt == "" or txt.lower() in {"na", "nan", "null", "none", "missing", "."}:
            return np.nan
        try:
            return float(txt)
        except (TypeError, ValueError):
            return np.nan

    def _fmt_pct(numer: int, denom: int) -> str:
        if denom == 0:
            return "NA"
        return f"{(100.0 * numer / denom):.2f}%"

    def _is_binary_phenotype(vals: np.ndarray) -> bool:
        finite = vals[~np.isnan(vals)]
        if finite.size == 0:
            return False
        uniq = np.unique(finite)
        return bool(np.all(np.isin(uniq, [0.0, 1.0])))

    def _mad(vals: np.ndarray) -> float:
        med = float(np.median(vals))
        return float(np.median(np.abs(vals - med)))

    sample_to_idx = {s: i for i, s in enumerate(samples)}
    pheno_vals = np.full(len(samples), np.nan)
    pheno_seen = np.zeros(len(samples), dtype=bool)
    pheno_cov_names: list[str] = []
    pheno_cov_vals: list[list[float]] = [[] for _ in samples]

    with open(pheno_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            logger.error("Phenotype file is empty or has no header")
            return 1
        pheno_cov_names = [
            c for c in reader.fieldnames if c not in ("sample_id", "phenotype")
        ]
        for row in reader:
            sid = row["sample_id"]
            if sid not in sample_to_idx:
                continue
            idx = sample_to_idx[sid]
            pheno_seen[idx] = True
            pheno_vals[idx] = _parse_float(row["phenotype"])
            for cn in pheno_cov_names:
                pheno_cov_vals[idx].append(_parse_float(row[cn]))

    valid_mask = ~np.isnan(pheno_vals)
    n_samples_total = len(samples)
    n_pheno_matched = int(pheno_seen.sum())
    n_valid_pre = int(valid_mask.sum())
    n_missing_pre = n_pheno_matched - n_valid_pre
    if n_valid_pre == 0:
        logger.error("No valid phenotype values matched to samples")
        return 1

    analyzed_mask = valid_mask.copy()
    n_hq_in_lrr = None
    n_dropped_lq = 0
    hq_source: str | None = None
    if args.hq_samples is not None:
        if not args.hq_samples.exists():
            logger.error("HQ sample list not found: %s", args.hq_samples)
            return 1
        hq_samples: set[str] = set()
        with open(args.hq_samples, newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                sid = row[0].strip()
                if not sid:
                    continue
                if sid.lower() in {"sample_id", "sample", "sampleid"}:
                    continue
                hq_samples.add(sid)
        hq_mask = np.array([s in hq_samples for s in samples], dtype=bool)
        n_hq_in_lrr = int(hq_mask.sum())
        n_dropped_lq = int((valid_mask & ~hq_mask).sum())
        analyzed_mask = valid_mask & hq_mask
        hq_source = "file"

        # Record sample QC in audit trail
        _hq_excluded: dict[str, str] = {
            s: "not_in_hq_list" for s in samples if s not in hq_samples
        }
        audit.record(
            stage="association_sample_qc",
            id_type="sample",
            included=[s for s in samples if s in hq_samples],
            excluded=_hq_excluded,
        )
    elif args.sample_sheet is not None:
        from array_lrr_gwas.sample_sheet import classify_samples_for_association

        # Resolve thresholds via apply_to_associate_args:
        # CLI --no-* flags → config → defaults.
        cli_overrides: dict[str, object] = {}
        if args.max_lrr_sd is not None:
            cli_overrides["max_lrr_sd"] = args.max_lrr_sd
        if args.min_sample_call_rate is not None:
            cli_overrides["min_call_rate"] = args.min_sample_call_rate
        if args.no_honor_precomputed:
            cli_overrides["honor_precomputed"] = False
        if args.no_exclude_baf_sd:
            cli_overrides["exclude_baf_sd"] = False
        if args.max_baf_sd is not None:
            cli_overrides["max_baf_sd"] = args.max_baf_sd
        if args.no_exclude_sex_discordant:
            cli_overrides["exclude_sex_discordant"] = False
        if args.no_exclude_extreme_inbreeding:
            cli_overrides["exclude_extreme_inbreeding"] = False
        if args.max_abs_inbreeding_f is not None:
            cli_overrides["max_abs_inbreeding_f"] = args.max_abs_inbreeding_f

        assoc_kwargs = apply_to_associate_args(cfg, cli_overrides)

        logger.info(
            "Deriving HQ samples from sample sheet %s "
            "(max_lrr_sd=%.4f, min_call_rate=%.4f, "
            "honor_precomputed=%s, exclude_baf_sd=%s [max=%.4f], "
            "exclude_sex_discordant=%s, exclude_extreme_inbreeding=%s [max|F|=%.4f])",
            args.sample_sheet,
            assoc_kwargs["max_lrr_sd"],
            assoc_kwargs["min_call_rate"],
            assoc_kwargs["honor_precomputed"],
            assoc_kwargs["exclude_baf_sd"],
            assoc_kwargs["max_baf_sd"],
            assoc_kwargs["exclude_sex_discordant"],
            assoc_kwargs["exclude_extreme_inbreeding"],
            assoc_kwargs["max_abs_inbreeding_f"],
        )
        excl_result = classify_samples_for_association(
            args.sample_sheet, **assoc_kwargs,
        )
        hq_samples = excl_result.hq_ids
        hq_mask = np.array([s in hq_samples for s in samples], dtype=bool)
        n_hq_in_lrr = int(hq_mask.sum())
        n_dropped_lq = int((valid_mask & ~hq_mask).sum())
        analyzed_mask = valid_mask & hq_mask
        hq_source = "sample_sheet"

        # Log per-category exclusion counts
        logger.info(
            "Sample exclusion summary (sheet total=%d): "
            "low_call_rate=%d, high_lrr_sd=%d, "
            "pre_pca_excluded=%d, excluded_relatedness=%d, "
            "excluded_het_outlier=%d, high_baf_sd=%d, "
            "sex_discordant=%d, extreme_inbreeding_f=%d, "
            "total_excluded=%d, passing=%d",
            excl_result.total,
            excl_result.counts["low_call_rate"],
            excl_result.counts["high_lrr_sd"],
            excl_result.counts["pre_pca_excluded"],
            excl_result.counts["excluded_relatedness"],
            excl_result.counts["excluded_het_outlier"],
            excl_result.counts["high_baf_sd"],
            excl_result.counts["sex_discordant"],
            excl_result.counts["extreme_inbreeding_f"],
            excl_result.counts["total_excluded"],
            excl_result.total - excl_result.counts["total_excluded"],
        )

        # Record sample exclusions in audit trail
        _sample_excluded: dict[str, str] = {}
        for sid, reasons in excl_result.excluded_reasons.items():
            _sample_excluded[sid] = ";".join(reasons)
        audit.record(
            stage="association_sample_qc",
            id_type="sample",
            included=list(excl_result.hq_ids),
            excluded=_sample_excluded,
        )

    n_analyzed = int(analyzed_mask.sum())
    if n_analyzed == 0:
        logger.error("No analyzable samples remain after filtering")
        return 1

    logger.info(
        "Association sample breakdown: total_lrr=%d, phenotype_matched=%d, "
        "phenotype_valid=%d, phenotype_missing=%d",
        n_samples_total, n_pheno_matched, n_valid_pre, n_missing_pre,
    )
    if n_hq_in_lrr is not None:
        logger.info(
            "Association HQ intersection (source=%s): hq_in_lrr=%d, "
            "valid_pheno_and_hq=%d, dropped_lq_with_valid_pheno=%d",
            hq_source, n_hq_in_lrr, n_analyzed, n_dropped_lq,
        )
    else:
        logger.info("Association analyzed samples: %d", n_analyzed)

    pre_vals = pheno_vals[valid_mask]
    analyzed_vals = pheno_vals[analyzed_mask]
    if _is_binary_phenotype(pre_vals):
        pre_cases = int(np.sum(pre_vals == 1.0))
        pre_controls = int(np.sum(pre_vals == 0.0))
        post_cases = int(np.sum(analyzed_vals == 1.0))
        post_controls = int(np.sum(analyzed_vals == 0.0))
        logger.info(
            "Phenotype summary (binary pre-filter): cases=%d (%s of valid, %s of total), "
            "controls=%d (%s of valid, %s of total), missing=%d (%s of matched)",
            pre_cases, _fmt_pct(pre_cases, n_valid_pre), _fmt_pct(pre_cases, n_samples_total),
            pre_controls, _fmt_pct(pre_controls, n_valid_pre), _fmt_pct(pre_controls, n_samples_total),
            n_missing_pre, _fmt_pct(n_missing_pre, n_pheno_matched),
        )
        logger.info(
            "Phenotype summary (binary analyzed): cases=%d (%s of analyzed, %s of total), "
            "controls=%d (%s of analyzed, %s of total)",
            post_cases, _fmt_pct(post_cases, n_analyzed), _fmt_pct(post_cases, n_samples_total),
            post_controls, _fmt_pct(post_controls, n_analyzed), _fmt_pct(post_controls, n_samples_total),
        )
    else:
        logger.info(
            "Phenotype summary (quantitative pre-filter): n=%d, mean=%.6g, median=%.6g, "
            "sd=%.6g, mad=%.6g, min=%.6g, max=%.6g",
            n_valid_pre,
            float(np.mean(pre_vals)),
            float(np.median(pre_vals)),
            float(np.std(pre_vals)),
            _mad(pre_vals),
            float(np.min(pre_vals)),
            float(np.max(pre_vals)),
        )
        logger.info(
            "Phenotype summary (quantitative analyzed): n=%d, mean=%.6g, median=%.6g, "
            "sd=%.6g, mad=%.6g, min=%.6g, max=%.6g",
            n_analyzed,
            float(np.mean(analyzed_vals)),
            float(np.median(analyzed_vals)),
            float(np.std(analyzed_vals)),
            _mad(analyzed_vals),
            float(np.min(analyzed_vals)),
            float(np.max(analyzed_vals)),
        )

    # Build covariate matrix
    cov_parts: list[np.ndarray] = []

    # Covariates from phenotype file
    if pheno_cov_names:
        pheno_covs = np.array(
            [pheno_cov_vals[i] for i in range(len(samples)) if analyzed_mask[i]]
        )
        cov_parts.append(pheno_covs)
        logger.info(
            "Using %d covariates from phenotype file: %s",
            len(pheno_cov_names), pheno_cov_names,
        )

    # Covariates from sample sheet
    if args.sample_sheet is not None:
        if not args.sample_sheet.exists():
            logger.error("Sample sheet not found: %s", args.sample_sheet)
            return 1

        from array_lrr_gwas.sample_sheet import read_sample_sheet, align_samples

        logger.info("Reading sample sheet from %s", args.sample_sheet)
        sheet_ids, sheet_covs, cov_names = read_sample_sheet(
            args.sample_sheet, n_pcs=args.n_pcs,
        )
        logger.info(
            "Extracted %d covariates from sample sheet: %s",
            len(cov_names), cov_names,
        )

        # Align to BCF samples and subset to valid
        aligned = align_samples(samples, sheet_ids, sheet_covs)
        aligned_valid = aligned[analyzed_mask]

        # Drop columns that are all NaN
        col_valid = ~np.all(np.isnan(aligned_valid), axis=0)
        if col_valid.any():
            aligned_valid = aligned_valid[:, col_valid]
            used_names = [n for n, v in zip(cov_names, col_valid) if v]
            # Mean-impute any remaining NaN (vectorised)
            nan_mask = np.isnan(aligned_valid)
            if nan_mask.any():
                col_means = np.nanmean(aligned_valid, axis=0, keepdims=True)
                aligned_valid = np.where(nan_mask, col_means, aligned_valid)
            cov_parts.append(aligned_valid)
            logger.info(
                "Using %d sample-sheet covariates: %s",
                len(used_names), used_names,
            )

    # Combine covariates
    covariates = None
    if cov_parts:
        covariates = np.column_stack(cov_parts)

    # Subset to samples with phenotype
    phenotype = pheno_vals[analyzed_mask]

    # ------------------------------------------------------------------
    # Association-stage marker exclusion (metadata-only pre-filter)
    # ------------------------------------------------------------------
    # INTENSITY_ONLY markers can be identified from metadata alone
    # (no LRR needed).  Monomorphic LRR filtering is deferred to the
    # streaming association scan because it requires loading LRR values.
    amqc = cfg.get("association_marker_qc", {})
    exclude_intensity_only = amqc.get("exclude_intensity_only", True)
    if args.no_exclude_intensity_only:
        exclude_intensity_only = False
    exclude_monomorphic = amqc.get("exclude_monomorphic_lrr", True)
    if args.no_exclude_monomorphic_lrr:
        exclude_monomorphic = False

    # Build a variant-level boolean mask for metadata-based exclusions.
    # This mask is passed to stream_lrr_chunks() so excluded variants
    # are never loaded into memory.
    n_total_markers = len(variants)
    variant_mask = np.ones(n_total_markers, dtype=bool)

    if exclude_intensity_only:
        intensity_mask = np.array(
            [v.get("intensity_only", False) for v in variants], dtype=bool,
        )
        n_intensity = int(intensity_mask.sum())
        variant_mask &= ~intensity_mask
        logger.info(
            "Metadata-based marker exclusion: INTENSITY_ONLY: %d / %d excluded "
            "(%s; non-polymorphic probes without GT)",
            n_intensity, n_total_markers,
            _fmt_pct(n_intensity, n_total_markers),
        )

    n_metadata_keep = int(variant_mask.sum())
    logger.info(
        "Metadata pre-filter: %d / %d markers pass (%d excluded, %s). "
        "Monomorphic LRR filtering deferred to streaming scan.",
        n_metadata_keep, n_total_markers,
        n_total_markers - n_metadata_keep,
        _fmt_pct(n_total_markers - n_metadata_keep, n_total_markers),
    )

    # Variants surviving the metadata filter — used for output provenance.
    # The full list of kept/excluded variants is populated during streaming.
    variants_meta_pass = [v for v, k in zip(variants, variant_mask) if k]

    # GRM for LMM
    grm = None
    if args.method == "lmm":
        from array_lrr_gwas.genotypes import read_genotypes
        from array_lrr_gwas.grm import compute_grm
        from array_lrr_gwas.variant_qc import read_collated_variant_qc, variant_qc_mask

        gt_path = args.genotype_bcf or input_path
        logger.info("Reading genotypes from %s for GRM", gt_path)
        dosage, gt_samples, gt_variants = read_genotypes(
            gt_path,
            min_maf=args.min_maf,
            min_call_rate=args.min_gt_call_rate,
        )

        if dosage.shape[0] == 0:
            logger.error(
                "No genotype variants passed QC filters from %s. "
                "Ensure FORMAT/GT field is present. "
                "Use --method ols to skip GRM requirement.",
                gt_path,
            )
            return 1

        n_initial = dosage.shape[0]
        logger.info(
            "%d genotype variants from %d samples passed initial QC",
            n_initial, dosage.shape[1],
        )

        # Apply upstream variant QC mask (call rate + HWE + MAF)
        variant_qc_path = args.variant_qc or cfg.get("upstream_qc", {}).get("variant_qc_path")
        if variant_qc_path is not None:
            logger.info(
                "Loading upstream variant QC from %s for GRM", variant_qc_path,
            )
            qc_data = read_collated_variant_qc(variant_qc_path)
            gt_variant_ids = [_variant_id(v) for v in gt_variants]
            qc_keep = variant_qc_mask(
                gt_variant_ids, qc_data,
                require_call_rate=True, require_hwe=True, require_maf=True,
                audit=audit, audit_stage="grm_variant_qc",
            )
            n_before_qc = dosage.shape[0]
            dosage = dosage[qc_keep]
            gt_variants = [v for v, k in zip(gt_variants, qc_keep) if k]
            logger.info(
                "Upstream variant QC (GRM): %d → %d variants "
                "(call rate + HWE + MAF required)",
                n_before_qc, dosage.shape[0],
            )

            if dosage.shape[0] == 0:
                logger.error(
                    "No genotype variants remain after upstream QC mask "
                    "(%d before filtering, file: %s). Check that variant "
                    "IDs match between the genotype file and "
                    "collated_variant_qc.tsv.",
                    n_before_qc, variant_qc_path,
                )
                return 1
        else:
            logger.warning(
                "No upstream variant QC file provided (--variant-qc or "
                "upstream_qc.variant_qc_path).  Skipping ancestry-informed "
                "marker filtering for GRM.  Provide collated_variant_qc.tsv "
                "for best-practice QC."
            )

        # LD prune GRM markers
        # Capture variants before pruning so we can report excluded IDs.
        _grm_pre_ld_variants = list(gt_variants)
        if not args.no_ld_prune:
            if args.ld_backend == "plink2":
                from array_lrr_gwas.ld_prune import ld_prune, ld_prune_plink2

                ld_window_kb = max(1, args.ld_window_bp // 1000)
                logger.info(
                    "LD pruning with plink2 (window=%dkb, r²=%.2f)",
                    ld_window_kb, args.ld_r2_thresh,
                )
                try:
                    keep_ids = ld_prune_plink2(
                        gt_path,
                        window_kb=ld_window_kb,
                        r2_thresh=args.ld_r2_thresh,
                        min_maf=args.min_maf,
                    )
                    # plink2 uses the VCF ID column; when ID is '.' it
                    # constructs chrom:pos:ref:alt.  Match on both forms.
                    ld_keep = np.array([
                        v.get("id") in keep_ids
                        or f"{v['chrom']}:{v['pos']}:{v.get('ref', '')}:{':'.join(v.get('alts', ()))}" in keep_ids
                        for v in gt_variants
                    ], dtype=bool)
                except FileNotFoundError:
                    logger.error(
                        "plink2 backend requested but plink2 is not on PATH. "
                        "Install plink2 (https://www.cog-genomics.org/plink/2.0/) "
                        "or explicitly select the NumPy fallback with "
                        "--ld-backend numpy."
                    )
                    return 1
            else:
                from array_lrr_gwas.ld_prune import ld_prune

                gt_positions = np.array(
                    [v["pos"] for v in gt_variants], dtype=np.intp
                )
                gt_chroms = np.array(
                    [v["chrom"] for v in gt_variants], dtype=str
                )
                logger.info(
                    "LD pruning with NumPy (window=%dbp, r²=%.2f)",
                    args.ld_window_bp, args.ld_r2_thresh,
                )
                ld_keep = ld_prune(
                    dosage,
                    positions=gt_positions,
                    chromosomes=gt_chroms,
                    window_bp=args.ld_window_bp,
                    r2_thresh=args.ld_r2_thresh,
                )

            n_before = dosage.shape[0]
            dosage = dosage[ld_keep]
            gt_variants = [v for v, k in zip(gt_variants, ld_keep) if k]
            logger.info(
                "LD pruning: %d → %d variants (removed %d)",
                n_before, dosage.shape[0], n_before - dosage.shape[0],
            )

            # Record GRM marker set after LD pruning in audit trail.
            _ld_included = [_variant_id(v) for v in gt_variants]
            _ld_excluded = {
                _variant_id(v): "ld_prune"
                for v, k in zip(_grm_pre_ld_variants, ld_keep) if not k
            }
            audit.record(
                stage="grm_ld_prune",
                id_type="marker",
                included=_ld_included,
                excluded=_ld_excluded,
            )

            if dosage.shape[0] == 0:
                logger.error(
                    "No variants remain after LD pruning.  Consider "
                    "relaxing --ld-r2-thresh or using --no-ld-prune."
                )
                return 1
        else:
            logger.info(
                "LD pruning disabled (--no-ld-prune).  All %d GRM markers "
                "retained without LD filtering.",
                dosage.shape[0],
            )
            # Record the full marker set (no exclusions) in audit trail.
            audit.record(
                stage="grm_ld_prune",
                id_type="marker",
                included=[_variant_id(v) for v in gt_variants],
                excluded={},
            )

        logger.info(
            "Using %d genotype variants from %d samples for GRM",
            dosage.shape[0], dosage.shape[1],
        )

        # Align GT samples to LRR samples
        gt_idx = {s: i for i, s in enumerate(gt_samples)}
        valid_samples = [samples[i] for i in range(len(samples)) if analyzed_mask[i]]
        gt_order = []
        for s in valid_samples:
            if s in gt_idx:
                gt_order.append(gt_idx[s])
            else:
                logger.error("Sample %s not found in genotype file", s)
                return 1

        # Record which samples are included in the GRM in audit trail.
        # Samples in the GT file that are not being analysed are excluded.
        _grm_sample_excluded = {
            s: "not_in_analyzed_set"
            for s in gt_samples if s not in set(valid_samples)
        }
        audit.record(
            stage="grm_samples",
            id_type="sample",
            included=valid_samples,
            excluded=_grm_sample_excluded,
        )

        dosage_aligned = dosage[:, gt_order]
        logger.info("Computing GRM...")
        grm = compute_grm(dosage_aligned, min_maf=args.min_maf)
        logger.info(
            "GRM computed: %d × %d", grm.shape[0], grm.shape[1],
        )

    if args.method == "logistic":
        logger.warning(
            "Logistic regression does not use the GRM random effect. "
            "Only fixed-effect covariates (e.g. PCs) are applied. "
            "For related samples with binary traits, consider using "
            "--method lmm as a continuous-trait approximation or "
            "pre-filtering highly related individuals."
        )

    # ------------------------------------------------------------------
    # Streaming association scan
    # ------------------------------------------------------------------
    # Stream LRR values from the BCF in chunks, applying per-chunk
    # marker filtering (monomorphic LRR) inline.  The full LRR matrix
    # is never held in memory.
    n_cov = covariates.shape[1] if covariates is not None else 0
    _grm_info = f", grm_size={grm.shape[0]}" if grm is not None else ""
    logger.info(
        "Running streaming %s association scan: n_markers_eligible=%d, "
        "n_samples=%d, n_covariates=%d%s",
        args.method, n_metadata_keep, int(analyzed_mask.sum()),
        n_cov, _grm_info,
    )

    lrr_stream = stream_lrr_chunks(
        input_path,
        chunk_size=5000,
        sample_mask=analyzed_mask,
        variant_mask=variant_mask,
    )
    result, exclusion_info = run_association_streaming(
        lrr_stream,
        phenotype,
        covariates=covariates,
        method=args.method,
        grm=grm,
        exclude_monomorphic=exclude_monomorphic,
        exclude_intensity_only=False,  # already excluded via variant_mask
    )
    n_tested = exclusion_info["n_tested"]
    n_mono_excluded = exclusion_info["n_monomorphic"]
    logger.info(
        "Streaming association complete: %d markers tested, "
        "%d monomorphic LRR excluded during scan (%s)",
        n_tested, n_mono_excluded,
        _fmt_pct(n_mono_excluded, exclusion_info["n_total"]),
    )

    # Record association marker exclusion in audit trail.
    # Combine metadata-based (INTENSITY_ONLY) and streaming (monomorphic)
    # exclusions into a single audit record.
    _all_marker_excluded: dict[str, str] = {}
    if exclude_intensity_only:
        for v, k in zip(variants, variant_mask):
            if not k:
                vid = _variant_id(v)
                if v.get("intensity_only", False):
                    _all_marker_excluded[vid] = "intensity_only"
    _all_marker_excluded.update(exclusion_info["excluded_markers"])
    _all_marker_included = [
        v.get("id") or f"{v['chrom']}:{v['pos']}"
        for v in result.to_records()
    ] if n_tested > 0 else []
    # Use variant_id from result directly
    _all_marker_included = list(result.variant_id)

    audit.record(
        stage="association_marker_exclusion",
        id_type="marker",
        included=_all_marker_included,
        excluded=_all_marker_excluded,
    )

    if n_tested == 0:
        logger.error(
            "No markers remain after association-stage filtering. "
            "Consider relaxing marker exclusion criteria."
        )
        return 1

    logger.info("Association complete for %d variants", len(result.chrom))

    # Log result summary statistics for auditing.
    if len(result.chrom) > 0:
        _valid_p_mask = ~np.isnan(result.p_value) & (result.p_value > 0)
        _valid_p = result.p_value[_valid_p_mask]
        _valid_stat = result.stat[~np.isnan(result.stat)]
        _valid_beta = result.beta[~np.isnan(result.beta)]
        _valid_se = result.se[~np.isnan(result.se)]
        if len(_valid_p) > 0:
            _n_gws = int(np.sum(_valid_p < 5e-8))
            _n_sug = int(np.sum(_valid_p < 1e-5))
            # Lambda GC: median(chi2) / 0.4549 using t-stat² as chi2(1) proxy.
            _lambda_gc = (
                float(np.median(_valid_stat ** 2) / 0.4549)
                if len(_valid_stat) > 0 else float("nan")
            )
            logger.info(
                "Result summary: n_tested=%d, min_p=%.3g, "
                "lambda_gc=%.4f, n_genome_wide_sig=%d (p<5e-8), "
                "n_suggestive=%d (p<1e-5)",
                len(result.chrom),
                float(np.min(_valid_p)),
                _lambda_gc,
                _n_gws,
                _n_sug,
            )
        if len(_valid_beta) > 0 and len(_valid_se) > 0:
            logger.info(
                "Effect size summary: beta [%.4g, %.4g] (mean_abs=%.4g), "
                "se [%.4g, %.4g]",
                float(np.min(_valid_beta)),
                float(np.max(_valid_beta)),
                float(np.mean(np.abs(_valid_beta))),
                float(np.min(_valid_se)),
                float(np.max(_valid_se)),
            )

    # Optionally load upstream variant QC flags for output provenance
    variant_qc_path = args.variant_qc or cfg.get("upstream_qc", {}).get("variant_qc_path")
    qc_provenance = None
    if variant_qc_path is not None:
        from array_lrr_gwas.variant_qc import read_collated_variant_qc as _read_vqc

        try:
            qc_provenance = _read_vqc(variant_qc_path)
            logger.info(
                "Propagating upstream QC flags to output TSV "
                "(%d QC records loaded)",
                len(qc_provenance),
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Could not load variant QC for output provenance: %s", exc,
            )

    # Write output
    logger.info("Writing results to %s", args.output)
    records = result.to_records()

    # Append QC provenance columns when upstream data is available
    _QC_COLS = (
        "all_ancestries_call_rate_pass",
        "all_ancestries_hwe_pass",
        "all_ancestries_maf_pass",
        "all_ancestries_qc_pass",
    )
    if qc_provenance is not None:
        lrr_variant_ids = list(result.variant_id)
        for rec, vid in zip(records, lrr_variant_ids):
            qc_rec = qc_provenance.get(vid)
            if qc_rec is not None:
                rec["in_variant_qc"] = True
                rec["all_ancestries_call_rate_pass"] = qc_rec.call_rate_pass
                rec["all_ancestries_hwe_pass"] = qc_rec.hwe_pass
                rec["all_ancestries_maf_pass"] = qc_rec.maf_pass
                rec["all_ancestries_qc_pass"] = (
                    ""
                    if qc_rec.qc_pass is None
                    else qc_rec.qc_pass
                )
            else:
                rec["in_variant_qc"] = False
                rec["all_ancestries_call_rate_pass"] = ""
                rec["all_ancestries_hwe_pass"] = ""
                rec["all_ancestries_maf_pass"] = ""
                rec["all_ancestries_qc_pass"] = ""

    # Append marker-exclusion provenance columns so users know which
    # markers were excluded (and why) without re-running.  For surviving
    # markers these will be False when default exclusions are on; when
    # --no-exclude-intensity-only or --no-exclude-monomorphic-lrr is
    # used, some True values may appear.
    # intensity_only is always False for tested markers (filtered upstream).
    for rec in records:
        rec["intensity_only"] = False
    # Use the monomorphic flags collected during streaming scan.
    _tested_mono = exclusion_info["tested_mono_flags"]
    for i, rec in enumerate(records):
        rec["lrr_monomorphic"] = bool(_tested_mono[i]) if i < len(_tested_mono) else False

    header = list(records[0].keys()) if records else []
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    # Write audit trail if requested
    audit_dir = getattr(args, "audit_dir", None)
    if audit_dir is not None:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit.write_tsv(audit_dir / "associate_audit.tsv")
        audit.write_json(audit_dir / "associate_audit.json")
        audit.write_summary_tsv(audit_dir / "associate_audit_summary.tsv")

    logger.info("Done.")
    return 0


def _run_segment(args: argparse.Namespace) -> int:
    """Execute the ``segment`` sub-command."""
    from array_lrr_gwas.segmentation import read_association_tsv, segment_associations

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    logger.info("Reading association results from %s", args.input)
    records = read_association_tsv(args.input)
    logger.info("Loaded %d marker records", len(records))

    # Build keyword arguments, only passing HMM params that were explicitly set.
    kwargs: dict[str, object] = dict(
        strategy=args.strategy,
        p_threshold=args.p_threshold,
        max_gap=args.max_gap,
        min_markers=args.min_markers,
    )
    if args.null_rate is not None:
        kwargs["null_rate"] = args.null_rate
    if args.signal_rate is not None:
        kwargs["signal_rate"] = args.signal_rate
    if args.prior_assoc is not None:
        kwargs["prior_assoc"] = args.prior_assoc
    if args.transition_prob is not None:
        kwargs["transition_prob"] = args.transition_prob

    result = segment_associations(records, **kwargs)

    logger.info("Writing %d segments to %s", len(result.chrom), args.output)
    result.write_bed(args.output)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
