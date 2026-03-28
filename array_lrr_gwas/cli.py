"""Command-line interface for LRR batch-effect correction.

Usage
-----
::

    array-lrr-gwas correct input.bcf -o corrected.bcf [--build GRCh38] [--k 5]
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
        description="Batch-effect correction for array-based LRR values.",
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


if __name__ == "__main__":
    sys.exit(main())
