#!/usr/bin/env python3
"""Convert NGS-PCA sample_qc.tsv to an array-lrr-gwas phenotype file.

The ``array-lrr-gwas associate`` command expects a tab-separated phenotype
file with at least two columns::

    sample_id   phenotype

This script reads the ``sample_qc.tsv`` produced by the NGS-PCA pipeline
(see ``docs/sample_qc_provenance.md``) and writes a minimal two-column
phenotype file using ``MTDNA_CN`` as the phenotype.

Covariates (e.g. sequencing depth, sex) are not included here; they should
be added separately via ``--sample-sheet`` when running
``array-lrr-gwas associate``.

The default input is the canonical upstream source fetched directly over
HTTPS — no local copy of ``sample_qc.tsv`` is required::

    python scripts/make_mtdna_cn_phenotype.py -o phenotype_mtdna_cn.tsv

A local file can be supplied instead::

    python scripts/make_mtdna_cn_phenotype.py path/to/sample_qc.tsv \\
        -o phenotype_mtdna_cn.tsv

See also
--------
* ``docs/sample_qc_provenance.md`` — provenance and column descriptions for
  the upstream ``sample_qc.tsv``.
* ``array-lrr-gwas associate --help`` — phenotype file format documentation.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical upstream source (NGS-PCA 1000G high-coverage example output,
# post PR #34 — SUPERPOPULATION and FAMILY_ROLE columns correctly populated).
_DEFAULT_URL = (
    "https://raw.githubusercontent.com/jlanej/NGS-PCA"
    "/refs/heads/master/example/1000G_highcov/output/qc_output/sample_qc.tsv"
)

# Column names in the NGS-PCA sample_qc.tsv (post PR #34 schema)
_SAMPLE_ID_COL = "SAMPLE_ID"
_PHENOTYPE_COL = "MTDNA_CN"

# Missing-value sentinels accepted as NA
_MISSING_VALUES = {"", "NA", "na", "NaN", "nan", "NULL", "null", "None", ".", "N/A"}


def _open_input(source: str) -> io.TextIOWrapper:
    """Return a text stream for *source*, which may be a URL or a file path."""
    if source.startswith("http://") or source.startswith("https://"):
        logger.info("Fetching remote file: %s", source)
        response = urllib.request.urlopen(source)
        return io.TextIOWrapper(response, encoding="utf-8")
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    logger.info("Reading local file: %s", path)
    return path.open(newline="", encoding="utf-8")


def convert(
    source: str,
    output_path: Path,
    *,
    verbose: bool = False,
) -> int:
    """Read *source* (URL or file path) and write a phenotype TSV to *output_path*.

    Parameters
    ----------
    source : str
        URL or local path to the NGS-PCA ``sample_qc.tsv``.
    output_path : Path
        Destination path for the two-column phenotype TSV
        (``sample_id`` / ``phenotype``).
    verbose : bool
        Enable debug logging.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    try:
        in_fh = _open_input(source)
    except (FileNotFoundError, Exception) as exc:
        logger.error("%s", exc)
        return 1

    n_written = 0
    n_missing_pheno = 0

    try:
        with in_fh, output_path.open("w", newline="", encoding="utf-8") as out_fh:
            reader = csv.DictReader(in_fh, delimiter="\t")
            if reader.fieldnames is None:
                logger.error("Input is empty or has no header: %s", source)
                return 1

            # Validate required columns
            required = {_SAMPLE_ID_COL, _PHENOTYPE_COL}
            missing_cols = required - set(reader.fieldnames)
            if missing_cols:
                logger.error(
                    "Input is missing required columns: %s",
                    sorted(missing_cols),
                )
                return 1

            writer = csv.DictWriter(
                out_fh,
                fieldnames=["sample_id", "phenotype"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()

            for row in reader:
                sample_id = row[_SAMPLE_ID_COL].strip()
                if not sample_id:
                    continue

                raw_pheno = row[_PHENOTYPE_COL].strip()
                phenotype = "NA" if raw_pheno in _MISSING_VALUES else raw_pheno

                if phenotype == "NA":
                    n_missing_pheno += 1

                writer.writerow({"sample_id": sample_id, "phenotype": phenotype})
                n_written += 1

    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        return 1

    n_valid = n_written - n_missing_pheno
    logger.info(
        "Wrote %d samples to %s  (phenotype: %d valid, %d missing/NA)",
        n_written,
        output_path,
        n_valid,
        n_missing_pheno,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_mtdna_cn_phenotype",
        description=(
            "Convert NGS-PCA sample_qc.tsv to an array-lrr-gwas phenotype "
            "file using MTDNA_CN as the phenotype."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=_DEFAULT_URL,
        metavar="SOURCE",
        help=(
            "URL or path to sample_qc.tsv produced by NGS-PCA.  "
            f"Defaults to the canonical upstream URL:\n{_DEFAULT_URL}"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        metavar="PHENOTYPE_TSV",
        help=(
            "Output path for the phenotype TSV "
            "(two columns: sample_id, phenotype)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
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

    return convert(
        args.input,
        args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
