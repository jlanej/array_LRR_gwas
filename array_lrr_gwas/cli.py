"""Command-line interface for LRR batch-effect correction and LMM association.

Usage
-----
::

    array-lrr-gwas correct input.bcf -o corrected.bcf [--build GRCh38] [--k 5]
    array-lrr-gwas associate input.bcf --phenotype pheno.tsv -o results.tsv
    array-lrr-gwas associate input.bcf --phenotype pheno.tsv \\
        --sample-sheet compiled_sample_sheet.tsv -o results.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
        help="Max per-sample LRR SD for HQ classification (default: 0.35).",
    )
    correct.add_argument(
        "--min-sample-call-rate",
        type=float,
        default=0.95,
        help="Min per-sample call rate for HQ classification (default: 0.95).",
    )
    correct.add_argument(
        "--min-marker-call-rate",
        type=float,
        default=0.95,
        help="Min per-marker call rate for subsetting (default: 0.95).",
    )
    correct.add_argument(
        "--min-var",
        type=float,
        default=0.001,
        help="Min per-marker variance (default: 0.001).",
    )
    correct.add_argument(
        "--max-var",
        type=float,
        default=None,
        help="Max per-marker variance (default: no upper limit).",
    )
    correct.add_argument(
        "--backend",
        type=str,
        default="rsvd",
        choices=["rsvd", "fbpca"],
        help="Decomposition backend (default: rsvd).",
    )
    correct.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
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
        help="Minimum MAF for GRM genotype filtering (default: 0.01).",
    )
    assoc.add_argument(
        "--min-gt-call-rate",
        type=float,
        default=0.90,
        help="Minimum genotype call rate for GRM filtering (default: 0.90).",
    )
    assoc.add_argument(
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

    parser.print_help()
    return 1


def _run_correct(args: argparse.Namespace) -> int:
    """Execute the ``correct`` sub-command."""
    import numpy as np

    from array_lrr_gwas.correction import correct_lrr
    from array_lrr_gwas.genome_build import detect_build, get_exclusion_regions
    from array_lrr_gwas.io_vcf import read_lrr, write_corrected

    input_path = args.input
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

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
    if not args.no_complexity_filter:
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

    # Run correction
    logger.info("Running batch-effect correction (k=%s)", args.k or "auto")
    corrected, info = correct_lrr(
        lrr,
        positions=positions,
        chromosomes=chromosomes,
        k=args.k,
        max_lrr_sd=args.max_lrr_sd,
        min_sample_call_rate=args.min_sample_call_rate,
        min_marker_call_rate=args.min_marker_call_rate,
        min_var=args.min_var,
        max_var=args.max_var,
        exclude_regions=exclude_regions,
        backend=args.backend,
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

    input_path = args.input
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    pheno_path = args.phenotype
    if not pheno_path.exists():
        logger.error("Phenotype file not found: %s", pheno_path)
        return 1

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

        gt_path = args.genotype_bcf or input_path
        logger.info("Reading genotypes from %s for GRM", gt_path)
        dosage, gt_samples, _ = read_genotypes(
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
            "Logistic regression does not use the GRM for relatedness "
            "correction.  Only fixed-effect covariates (e.g. PCs) are "
            "applied.  Consider pre-filtering highly related individuals "
            "or using --method lmm with a continuous phenotype proxy."
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


if __name__ == "__main__":
    sys.exit(main())
