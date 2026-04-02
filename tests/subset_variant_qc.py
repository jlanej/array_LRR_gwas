#!/usr/bin/env python3
"""Subset collated_variant_qc.tsv to markers present in a BCF/VCF file.

The full ``collated_variant_qc.tsv`` covers all array markers (~2.3 M rows).
For the 100-sample, 12 109-variant test BCF we only need the rows that match
those markers, keeping the file small enough to commit and fast to read.

The output ``test_variant_qc.tsv`` is a drop-in replacement for the full
file in the ``--variant-qc`` argument of ``array-lrr-gwas correct`` and
``array-lrr-gwas associate``.

Usage
-----
::

    python tests/subset_variant_qc.py          # uses built-in default paths
    python tests/subset_variant_qc.py \\
        --bcf   tests/data/stage2_reclustered.100.subsample.subset.bcf \\
        --input tests/data/collated_variant_qc.tsv \\
        --output tests/data/test_variant_qc.tsv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_BCF = (
    _REPO_ROOT / "tests" / "data" / "stage2_reclustered.100.subsample.subset.bcf"
)
_DEFAULT_INPUT = _REPO_ROOT / "tests" / "data" / "collated_variant_qc.tsv"
_DEFAULT_OUTPUT = _REPO_ROOT / "tests" / "data" / "test_variant_qc.tsv"

_QC_ID_COL = "variant_id"


def _bcf_variant_ids(bcf_path: Path) -> set[str]:
    """Return the set of all variant IDs in *bcf_path* using pysam."""
    try:
        import pysam
    except ImportError:
        logger.error("pysam is required — run: pip install pysam")
        sys.exit(1)

    ids: set[str] = set()
    with pysam.VariantFile(str(bcf_path)) as vcf:
        for rec in vcf:
            vid = rec.id
            if vid and vid != ".":
                ids.add(vid)
    logger.info("Extracted %d variant IDs from %s", len(ids), bcf_path)
    return ids


def subset(
    bcf_path: Path,
    input_path: Path,
    output_path: Path,
) -> int:
    """Write rows from *input_path* whose ``variant_id`` is in *bcf_path*.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    if not bcf_path.exists():
        logger.error("BCF not found: %s", bcf_path)
        return 1
    if not input_path.exists():
        logger.error("Input QC file not found: %s", input_path)
        return 1

    # 1. Collect all variant IDs from the BCF
    bcf_ids = _bcf_variant_ids(bcf_path)
    if not bcf_ids:
        logger.error("No variant IDs found in %s", bcf_path)
        return 1

    # 2. Stream through the QC file and keep matching rows
    n_written = 0
    n_total = 0

    with (
        input_path.open(newline="", encoding="utf-8") as in_fh,
        output_path.open("w", newline="", encoding="utf-8") as out_fh,
    ):
        reader = csv.DictReader(in_fh, delimiter="\t")
        if reader.fieldnames is None:
            logger.error("Input QC file is empty or has no header")
            return 1

        if _QC_ID_COL not in reader.fieldnames:
            logger.error(
                "Required column '%s' not found in %s", _QC_ID_COL, input_path
            )
            return 1

        writer = csv.DictWriter(
            out_fh,
            fieldnames=reader.fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for row in reader:
            n_total += 1
            if row[_QC_ID_COL] in bcf_ids:
                writer.writerow(row)
                n_written += 1

    n_missing = len(bcf_ids) - n_written
    logger.info(
        "Wrote %d / %d BCF markers to %s  "
        "(%d BCF markers had no QC entry; scanned %d QC rows)",
        n_written, len(bcf_ids), output_path, n_missing, n_total,
    )
    if n_missing:
        logger.warning(
            "%d BCF marker(s) were not found in the QC file — "
            "they will be unfiltered by --variant-qc",
            n_missing,
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subset_variant_qc",
        description=(
            "Subset collated_variant_qc.tsv to markers present in a BCF/VCF."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bcf",
        type=Path,
        default=_DEFAULT_BCF,
        metavar="BCF",
        help=f"Input BCF/VCF file.  Default: {_DEFAULT_BCF}",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        metavar="TSV",
        help=f"Full collated_variant_qc.tsv.  Default: {_DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="TSV",
        help=f"Output subset TSV.  Default: {_DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    return subset(args.bcf, args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())

