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
            "(default: no limit). When the QC-passing marker set would "
            "exceed this budget, markers are deterministically subsampled "
            "to fit. Genome-uniform sampling ensures no region is over- or "
            "under-represented. A value of 0 disables subsampling entirely."
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
            "faster pruning on large datasets; if unavailable, the "
            "CLI falls back to the NumPy backend with a warning."
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
    from array_lrr_gwas.io_vcf import read_lrr, write_corrected
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
    correct_kwargs["max_ram_gb"] = max_ram_gb

    # Read input
    logger.info("Reading LRR from %s", input_path)
    lrr, samples, variants = read_lrr(input_path)
    logger.info(
        "Loaded %d variants × %d samples", lrr.shape[0], lrr.shape[1]
    )

    # Extract positions and chromosomes from variants
    positions = np.array([v["pos"] for v in variants], dtype=np.intp)
    chromosomes = np.array([v["chrom"] for v in variants], dtype=str)

    # Determine exclusion regions
    exclude_regions = None
    no_complexity = (
        args.no_complexity_filter
        or cfg["correction"].get("no_complexity_filter", False)
    )
    if not no_complexity:
        build = args.build
        if build is None:
            build = detect_build(input_path)
        if build is None:
            logger.error(
                "Could not detect genome build from input file. "
                "Please supply --build (GRCh37, GRCh38, or T2T-CHM13)."
            )
            return 1
        logger.info("Using genome build: %s", build)
        exclude_regions = get_exclusion_regions(
            build, chromosomes=list(set(chromosomes))
        )
        n_regions = sum(len(v) for v in exclude_regions.values())
        logger.info(
            "Applying %d default exclusion regions (%s)",
            n_regions,
            build,
        )

    # Load upstream variant QC mask for RSVD marker selection
    variant_qc_path = args.variant_qc or cfg.get("upstream_qc", {}).get("variant_qc_path")
    upstream_qc_mask = None
    if variant_qc_path is not None:
        variant_qc_path = Path(variant_qc_path)
        logger.info("Loading upstream variant QC from %s", variant_qc_path)
        qc_data = read_collated_variant_qc(variant_qc_path)
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
    else:
        logger.warning(
            "No upstream variant QC file provided (--variant-qc or "
            "upstream_qc.variant_qc_path).  Skipping ancestry-informed "
            "marker filtering for RSVD.  Provide collated_variant_qc.tsv "
            "for best-practice QC."
        )

    # Upfront RAM estimation: use sample count and estimated variant count
    # from the QC sheet to warn early if subsampling will likely be needed.
    if max_ram_gb is not None and max_ram_gb > 0:
        from array_lrr_gwas.decomposition import estimate_rsvd_marker_budget

        n_samples_est = len(samples)
        # Use QC-passing variant count as the best available estimate
        # of how many markers will enter the RSVD; fall back to total
        # variant count when no QC sheet is available.
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

    logger.info("Done.")
    return 0


def _run_associate(args: argparse.Namespace) -> int:
    """Execute the ``associate`` sub-command."""
    import csv
    import warnings

    import numpy as np

    from array_lrr_gwas.audit import AuditLogger
    from array_lrr_gwas.association import run_association
    from array_lrr_gwas.io_vcf import read_lrr
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

    # Read LRR
    logger.info("Reading LRR from %s", input_path)
    lrr, samples, variants = read_lrr(input_path)
    logger.info(
        "Loaded %d variants × %d samples", lrr.shape[0], lrr.shape[1]
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
    lrr_sub = lrr[:, analyzed_mask]

    # ------------------------------------------------------------------
    # Association-stage marker exclusion
    # ------------------------------------------------------------------
    # Only two hard-fail pre-filters for LRR association markers:
    #   1. INTENSITY_ONLY (no cluster model → unreliable LRR)
    #   2. Monomorphic LRR (zero variance → degenerate statistics)
    # Upstream variant QC flags are NOT used to pre-filter — they are
    # propagated to the output TSV for post-hoc filtering.
    amqc = cfg.get("association_marker_qc", {})
    exclude_intensity_only = amqc.get("exclude_intensity_only", True)
    if args.no_exclude_intensity_only:
        exclude_intensity_only = False
    exclude_monomorphic = amqc.get("exclude_monomorphic_lrr", True)
    if args.no_exclude_monomorphic_lrr:
        exclude_monomorphic = False

    n_total_markers = lrr_sub.shape[0]
    marker_keep = np.ones(n_total_markers, dtype=bool)

    # 1. Exclude INTENSITY_ONLY markers
    if exclude_intensity_only:
        intensity_mask = np.array(
            [v.get("intensity_only", False) for v in variants], dtype=bool,
        )
        n_intensity = int(intensity_mask.sum())
        marker_keep &= ~intensity_mask
        logger.info(
            "Association marker exclusion: INTENSITY_ONLY: %d / %d excluded "
            "(non-polymorphic probes without GT; retained for correction only)",
            n_intensity, n_total_markers,
        )

    # 2. Exclude monomorphic LRR markers (zero variance across samples)
    # Always compute variance for provenance; optionally exclude.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _pre_filter_lrr_var = np.nanvar(lrr_sub, axis=1)
    _pre_filter_all_nan = np.all(np.isnan(lrr_sub), axis=1)
    _pre_filter_mono = (_pre_filter_lrr_var == 0.0) | _pre_filter_all_nan

    if exclude_monomorphic:
        zero_var_mask = (_pre_filter_lrr_var == 0.0) & ~_pre_filter_all_nan
        mono_mask = zero_var_mask | _pre_filter_all_nan
        n_mono = int(mono_mask.sum())
        n_zero_var = int(zero_var_mask.sum())
        n_all_nan = int(_pre_filter_all_nan.sum())
        marker_keep &= ~mono_mask
        logger.info(
            "Association marker exclusion: monomorphic LRR "
            "(zero variance): %d / %d excluded "
            "(%d constant, %d all-NaN)",
            n_mono, n_total_markers, n_zero_var, n_all_nan,
        )

    n_keep = int(marker_keep.sum())
    n_excluded = n_total_markers - n_keep
    logger.info(
        "Association marker exclusion summary: %d / %d markers pass all "
        "filters (%d excluded, %.1f%%)",
        n_keep, n_total_markers, n_excluded,
        100.0 * n_excluded / n_total_markers if n_total_markers > 0 else 0.0,
    )

    # Record association marker exclusion in audit trail
    _marker_excluded: dict[str, str] = {}
    _marker_included: list[str] = []
    for i, (v, keep) in enumerate(zip(variants, marker_keep)):
        vid = _variant_id(v)
        if keep:
            _marker_included.append(vid)
        else:
            reasons = []
            if exclude_intensity_only and v.get("intensity_only", False):
                reasons.append("intensity_only")
            if exclude_monomorphic and _pre_filter_mono[i]:
                reasons.append("monomorphic_lrr")
            _marker_excluded[vid] = ";".join(reasons) if reasons else "excluded"
    audit.record(
        stage="association_marker_exclusion",
        id_type="marker",
        included=_marker_included,
        excluded=_marker_excluded,
    )

    if n_keep == 0:
        logger.error(
            "No markers remain after association-stage filtering. "
            "Consider relaxing marker exclusion criteria."
        )
        return 1

    # Apply marker filter and carry forward the pre-computed monomorphic
    # flags for surviving markers (avoids recomputing variance later).
    _post_filter_mono = _pre_filter_mono[marker_keep]
    lrr_sub = lrr_sub[marker_keep]
    variants = [v for v, k in zip(variants, marker_keep) if k]

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
                    logger.warning(
                        "plink2 backend requested but plink2 is not on PATH; "
                        "falling back to NumPy LD pruning."
                    )
                    gt_positions = np.array(
                        [v["pos"] for v in gt_variants], dtype=np.intp
                    )
                    gt_chroms = np.array(
                        [v["chrom"] for v in gt_variants], dtype=str
                    )
                    ld_keep = ld_prune(
                        dosage,
                        positions=gt_positions,
                        chromosomes=gt_chroms,
                        window_bp=args.ld_window_bp,
                        r2_thresh=args.ld_r2_thresh,
                    )
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

    # Run association
    logger.info("Running %s association scan", args.method)
    result = run_association(
        lrr_sub, phenotype, variants,
        covariates=covariates,
        method=args.method,
        grm=grm,
    )
    logger.info("Association complete for %d variants", len(result.chrom))

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
        lrr_variant_ids = [_variant_id(v) for v in variants]
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
    for rec, v in zip(records, variants):
        rec["intensity_only"] = v.get("intensity_only", False)
    # Use the pre-computed monomorphic flags from the exclusion step
    # (cached in _post_filter_mono) to avoid recomputing variance.
    for i, rec in enumerate(records):
        rec["lrr_monomorphic"] = bool(_post_filter_mono[i])

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
