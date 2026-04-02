#!/usr/bin/env python3
"""Build the phenotype + covariate file used for pipeline testing.

Combines two sources:

1. **Phenotype** — ``MTDNA_CN`` (mitochondrial DNA copy number) fetched from
   the NGS-PCA 1000G high-coverage example output.  See
   ``tests/make_mtdna_cn_phenotype.py`` and
   ``docs/sample_qc_provenance.md`` for full provenance.

2. **Covariates** — from ``tests/data/compiled_sample_sheet.tsv``:

   * ``sex_numeric`` — sex encoded as ``1`` (M) / ``0`` (F) from the
     ``computed_gender`` column.  Biological sex is a strong confounder for
     mtDNA-CN (females typically have higher CN) and must be included.
   * ``PC1`` – ``PC20`` — global ancestry principal components from the
     Illumina IDAT processing pipeline.  These control for population
     stratification in the LRR-based GWAS.

Output is a single tab-separated file::

    sample_id  phenotype  sex_numeric  PC1  PC2  …  PC20

which is the format expected by ``array-lrr-gwas associate --phenotype``.
Any additional columns present are treated as fixed-effect covariates.

Usage
-----
::

    # Write to default path (tests/data/test_phenotype.tsv):
    python tests/make_test_phenotype.py

    # Explicit output path:
    python tests/make_test_phenotype.py -o /path/to/phenotype.tsv

    # Use a local sample_qc.tsv instead of fetching from URL:
    python tests/make_test_phenotype.py --qc-source /path/to/sample_qc.tsv

See also
--------
* ``docs/sample_qc_provenance.md`` — column descriptions for the upstream
  NGS-PCA QC file.
* ``tests/make_mtdna_cn_phenotype.py`` — minimal two-column version (no
  covariates) sourcing only from the upstream URL.
* ``scripts/run_correction.sh`` — runs ``array-lrr-gwas correct`` on the
  100-sample test BCF.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_QC_URL = (
    "https://raw.githubusercontent.com/jlanej/NGS-PCA"
    "/refs/heads/master/example/1000G_highcov/output/qc_output/sample_qc.tsv"
)

_DEFAULT_SAMPLE_SHEET = Path(__file__).parent / "data" / "compiled_sample_sheet.tsv"
_DEFAULT_OUTPUT = Path(__file__).parent / "data" / "test_phenotype.tsv"

# Columns sourced from the upstream NGS-PCA QC file
_QC_SAMPLE_COL = "SAMPLE_ID"
_QC_PHENO_COL = "MTDNA_CN"

# Columns sourced from the compiled sample sheet
_SHEET_SAMPLE_COL = "sample_id"
_SHEET_SEX_COL = "computed_gender"
_N_PCS = 20  # PC1 … PC20 (global ancestry)

# Missing-value sentinels
_MISSING = {"", "NA", "na", "NaN", "nan", "NULL", "null", "None", ".", "N/A"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_source(source: str) -> io.TextIOWrapper:
    """Return a text stream from *source* (URL or local path)."""
    if source.startswith("http://") or source.startswith("https://"):
        logger.info("Fetching: %s", source)
        return io.TextIOWrapper(urllib.request.urlopen(source), encoding="utf-8")
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    logger.info("Reading: %s", p)
    return p.open(newline="", encoding="utf-8")


def _encode_sex(val: str | None) -> str:
    """Return ``'1'`` (male), ``'0'`` (female), or ``'NA'``."""
    if not val or val.strip() in _MISSING:
        return "NA"
    v = val.strip().upper()
    if v == "M":
        return "1"
    if v == "F":
        return "0"
    return "NA"


def _clean(val: str | None) -> str:
    """Replace recognised missing-value strings with ``'NA'``."""
    if val is None:
        return "NA"
    s = val.strip()
    return "NA" if s in _MISSING else s


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build(
    qc_source: str,
    sheet_path: Path,
    output_path: Path,
) -> int:
    """Merge phenotype and covariates and write the combined TSV.

    Parameters
    ----------
    qc_source : str
        URL or local path to the NGS-PCA ``sample_qc.tsv``.
    sheet_path : Path
        Path to ``compiled_sample_sheet.tsv``.
    output_path : Path
        Destination for the combined phenotype + covariate TSV.

    Returns
    -------
    int
        Exit code (0 = success, non-zero = error).
    """
    # ------------------------------------------------------------------
    # 1. Load MTDNA_CN from upstream QC file
    # ------------------------------------------------------------------
    try:
        qc_fh = _open_source(qc_source)
    except Exception as exc:
        logger.error("Could not open QC source: %s", exc)
        return 1

    mtdna: dict[str, str] = {}
    with qc_fh:
        reader = csv.DictReader(qc_fh, delimiter="\t")
        if reader.fieldnames is None:
            logger.error("QC source is empty or has no header")
            return 1
        missing_cols = {_QC_SAMPLE_COL, _QC_PHENO_COL} - set(reader.fieldnames)
        if missing_cols:
            logger.error("QC source missing required columns: %s", sorted(missing_cols))
            return 1
        for row in reader:
            sid = row[_QC_SAMPLE_COL].strip()
            if sid:
                mtdna[sid] = _clean(row[_QC_PHENO_COL])

    logger.info("Loaded MTDNA_CN for %d samples from QC source", len(mtdna))

    # ------------------------------------------------------------------
    # 2. Load covariates from compiled sample sheet
    # ------------------------------------------------------------------
    if not sheet_path.exists():
        logger.error("Sample sheet not found: %s", sheet_path)
        return 1

    pc_cols = [f"PC{i}" for i in range(1, _N_PCS + 1)]
    out_fields = ["sample_id", "phenotype", "sex_numeric"] + pc_cols

    n_written = 0
    n_missing_pheno = 0
    n_missing_sheet = 0

    with (
        sheet_path.open(newline="", encoding="utf-8") as sheet_fh,
        output_path.open("w", newline="", encoding="utf-8") as out_fh,
    ):
        reader = csv.DictReader(sheet_fh, delimiter="\t")
        if reader.fieldnames is None:
            logger.error("Sample sheet is empty or has no header")
            return 1

        # Validate sample sheet columns
        sheet_cols = set(reader.fieldnames)
        required_sheet = {_SHEET_SAMPLE_COL, _SHEET_SEX_COL} | set(pc_cols)
        missing_sheet_cols = required_sheet - sheet_cols
        if missing_sheet_cols:
            logger.error(
                "Sample sheet missing required columns: %s",
                sorted(missing_sheet_cols),
            )
            return 1

        writer = csv.DictWriter(
            out_fh, fieldnames=out_fields, delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()

        for row in reader:
            sid = row[_SHEET_SAMPLE_COL].strip()
            if not sid:
                continue

            if sid not in mtdna:
                n_missing_sheet += 1
                logger.debug("Sample %s not in QC source — skipping", sid)
                continue

            phenotype = mtdna[sid]
            if phenotype == "NA":
                n_missing_pheno += 1

            out_row: dict[str, str] = {
                "sample_id": sid,
                "phenotype": phenotype,
                "sex_numeric": _encode_sex(row.get(_SHEET_SEX_COL)),
            }
            for pc in pc_cols:
                out_row[pc] = _clean(row.get(pc))

            writer.writerow(out_row)
            n_written += 1

    n_valid = n_written - n_missing_pheno
    logger.info(
        "Wrote %d samples to %s  "
        "(phenotype: %d valid, %d missing/NA; "
        "%d sample-sheet samples lacked QC phenotype)",
        n_written, output_path, n_valid, n_missing_pheno, n_missing_sheet,
    )
    logger.info(
        "Covariates written: sex_numeric, %s",
        ", ".join(pc_cols),
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_test_phenotype",
        description=(
            "Build the combined phenotype + covariate file for pipeline "
            "testing (MTDNA_CN phenotype + sex + global PCs)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--qc-source",
        default=_DEFAULT_QC_URL,
        metavar="URL_OR_PATH",
        help=(
            "URL or local path to NGS-PCA sample_qc.tsv (source of MTDNA_CN). "
            f"Default: {_DEFAULT_QC_URL}"
        ),
    )
    parser.add_argument(
        "--sample-sheet",
        type=Path,
        default=_DEFAULT_SAMPLE_SHEET,
        metavar="TSV",
        help=(
            "Path to compiled_sample_sheet.tsv (source of covariates). "
            f"Default: {_DEFAULT_SAMPLE_SHEET}"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="TSV",
        help=(
            "Output path for the combined phenotype + covariate TSV. "
            f"Default: {_DEFAULT_OUTPUT}"
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

    return build(
        qc_source=args.qc_source,
        sheet_path=args.sample_sheet,
        output_path=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())

