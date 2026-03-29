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

logger = logging.getLogger(__name__)


def _variant_id(v: dict) -> str:
    """Build a canonical variant ID string from a variant metadata dict."""
    vid = v.get("id")
    if vid is not None and vid != ".":
        return vid
    alts = v.get("alts") or ()
    return f"{v['chrom']}:{v['pos']}:{v.get('ref', '')}:{':'.join(alts)}"


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
            "Number of batch-effect components to remove. "
            "Auto-selected via Marchenko–Pastur heuristic if omitted."
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
            "required for batch correction). Also configurable via "
            "upstream_qc.variant_qc_path in the YAML config."
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
        "--n-pcs",
        type=int,
        default=20,
        help="Number of global ancestry PCs to use from sample sheet (default: 20).",
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
            "are LD-pruned so that highly linked regions do not "
            "disproportionately dominate the GRM eigenstructure."
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
            "for GRM computation."
        ),
    )
    assoc.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML configuration file. For association, this is currently used "
            "to read upstream_qc.variant_qc_path when --variant-qc is not set."
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
    import numpy as np

    from array_lrr_gwas.correction import correct_lrr
    from array_lrr_gwas.genome_build import detect_build, get_exclusion_regions
    from array_lrr_gwas.io_vcf import read_lrr, write_corrected
    from array_lrr_gwas.qc_config import defaults, load_config, apply_to_correct_args
    from array_lrr_gwas.variant_qc import read_collated_variant_qc, variant_qc_mask

    input_path = args.input
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

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
        "backend": "rsvd",
    }
    for key, default_val in parser_defaults.items():
        val = getattr(args, key, None)
        if val != default_val:
            cli_overrides[key] = val

    correct_kwargs = apply_to_correct_args(cfg, cli_overrides)

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
        )
        n_pass = int(upstream_qc_mask.sum())
        logger.info(
            "Upstream variant QC (RSVD): %d / %d markers pass "
            "(call rate + HWE; MAF not required)",
            n_pass, len(upstream_qc_mask),
        )

    # Run correction
    logger.info(
        "Running batch-effect correction (k=%s)",
        correct_kwargs.get("k") or "auto",
    )
    corrected, info = correct_lrr(
        lrr,
        positions=positions,
        chromosomes=chromosomes,
        exclude_regions=exclude_regions,
        upstream_qc_mask=upstream_qc_mask,
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
    logger.info("Done.")
    return 0


def _run_associate(args: argparse.Namespace) -> int:
    """Execute the ``associate`` sub-command."""
    import csv

    import numpy as np

    from array_lrr_gwas.association import run_association
    from array_lrr_gwas.io_vcf import read_lrr
    from array_lrr_gwas.qc_config import defaults, load_config

    input_path = args.input
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    pheno_path = args.phenotype
    if not pheno_path.exists():
        logger.error("Phenotype file not found: %s", pheno_path)
        return 1

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
    sample_to_idx = {s: i for i, s in enumerate(samples)}
    pheno_vals = np.full(len(samples), np.nan)
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
            pheno_vals[idx] = float(row["phenotype"])
            for cn in pheno_cov_names:
                pheno_cov_vals[idx].append(float(row[cn]))

    valid_mask = ~np.isnan(pheno_vals)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        logger.error("No valid phenotype values matched to samples")
        return 1
    logger.info("%d samples with valid phenotype", n_valid)

    # Build covariate matrix
    cov_parts: list[np.ndarray] = []

    # Covariates from phenotype file
    if pheno_cov_names:
        pheno_covs = np.array(
            [pheno_cov_vals[i] for i in range(len(samples)) if valid_mask[i]]
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
        aligned_valid = aligned[valid_mask]

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
    phenotype = pheno_vals[valid_mask]
    lrr_sub = lrr[:, valid_mask]

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

        logger.info(
            "Using %d genotype variants from %d samples for GRM",
            dosage.shape[0], dosage.shape[1],
        )

        # Align GT samples to LRR samples
        gt_idx = {s: i for i, s in enumerate(gt_samples)}
        valid_samples = [samples[i] for i in range(len(samples)) if valid_mask[i]]
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

    # Write output
    logger.info("Writing results to %s", args.output)
    records = result.to_records()
    header = list(records[0].keys()) if records else []
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

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
