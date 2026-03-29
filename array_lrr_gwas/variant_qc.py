"""Loader and mask for upstream collated variant QC metrics.

Reads ``collated_variant_qc.tsv`` produced by the upstream
``jlanej/illumina_idat_processing`` pipeline and builds boolean masks
that downstream batch correction, GRM, and association steps can use
to apply ancestry-informed, best-practice marker QC without code
duplication.

Expected TSV columns
--------------------
* ``variant_id`` — unique marker identifier (e.g. ``chr1:12345:A:T``).
* ``all_ancestries_call_rate_pass`` — ``True``/``False`` flag indicating
  the marker's call rate passes the threshold across **all** ancestry
  strata.
* ``all_ancestries_hwe_pass`` — ``True``/``False`` Hardy-Weinberg
  equilibrium flag across all ancestries.
* ``all_ancestries_maf_pass`` — ``True``/``False`` minor-allele-frequency
  flag across all ancestries.
"""

from __future__ import annotations

import csv
import logging
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Column names expected in the upstream TSV.
_COL_VARIANT_ID = "variant_id"
_COL_CALL_RATE = "all_ancestries_call_rate_pass"
_COL_HWE = "all_ancestries_hwe_pass"
_COL_MAF = "all_ancestries_maf_pass"
_REQUIRED_COLUMNS = frozenset({_COL_VARIANT_ID, _COL_CALL_RATE, _COL_HWE, _COL_MAF})

# Recognised boolean true literals (case-insensitive).
_TRUE_LITERALS = frozenset({"true", "1", "yes"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_bool(value: str) -> bool:
    """Parse a string as a boolean flag."""
    return value.strip().lower() in _TRUE_LITERALS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class VariantQCRecord:
    """Lightweight container for a single variant's QC flags."""

    __slots__ = ("variant_id", "call_rate_pass", "hwe_pass", "maf_pass")

    def __init__(
        self,
        variant_id: str,
        call_rate_pass: bool,
        hwe_pass: bool,
        maf_pass: bool,
    ) -> None:
        self.variant_id = variant_id
        self.call_rate_pass = call_rate_pass
        self.hwe_pass = hwe_pass
        self.maf_pass = maf_pass


def read_collated_variant_qc(
    path: str | Path,
) -> dict[str, VariantQCRecord]:
    """Parse an upstream ``collated_variant_qc.tsv`` file.

    Parameters
    ----------
    path : str or Path
        Path to the tab-separated QC file.

    Returns
    -------
    dict[str, VariantQCRecord]
        Mapping from *variant_id* to its QC record.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the TSV is missing required columns or is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Variant QC file not found: {path}")

    records: dict[str, VariantQCRecord] = {}
    duplicates: list[str] = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError(f"Variant QC file is empty or has no header: {path}")

        present_columns = frozenset(reader.fieldnames)
        missing_columns = _REQUIRED_COLUMNS - present_columns
        if missing_columns:
            raise ValueError(
                f"Variant QC file is missing required columns: "
                f"{sorted(missing_columns)}.  "
                f"Expected: {sorted(_REQUIRED_COLUMNS)}"
            )

        for row in reader:
            vid = row[_COL_VARIANT_ID]
            if vid in records:
                duplicates.append(vid)
                continue  # keep first occurrence
            records[vid] = VariantQCRecord(
                variant_id=vid,
                call_rate_pass=_parse_bool(row[_COL_CALL_RATE]),
                hwe_pass=_parse_bool(row[_COL_HWE]),
                maf_pass=_parse_bool(row[_COL_MAF]),
            )

    if duplicates:
        n = len(duplicates)
        sample = duplicates[:5]
        warnings.warn(
            f"{n} duplicate variant_id(s) in {path.name}; "
            f"kept first occurrence.  Examples: {sample}",
            stacklevel=2,
        )

    if not records:
        raise ValueError(f"Variant QC file contains no data rows: {path}")

    logger.info("Loaded %d variant QC records from %s", len(records), path.name)
    return records


def variant_qc_mask(
    variant_ids: Sequence[str] | NDArray,
    qc_data: dict[str, VariantQCRecord] | None = None,
    *,
    require_call_rate: bool = True,
    require_hwe: bool = True,
    require_maf: bool = False,
) -> NDArray[np.bool_]:
    """Build a boolean keep-mask aligned to *variant_ids*.

    Parameters
    ----------
    variant_ids : sequence of str
        Ordered variant identifiers (e.g. from the input VCF/BCF).
    qc_data : dict or None
        Output of :func:`read_collated_variant_qc`.  If ``None``, a
        fallback all-``True`` mask is returned with a warning.
    require_call_rate : bool
        If ``True``, variants failing ``all_ancestries_call_rate_pass``
        are masked out.
    require_hwe : bool
        If ``True``, variants failing ``all_ancestries_hwe_pass`` are
        masked out.
    require_maf : bool
        If ``True``, variants failing ``all_ancestries_maf_pass`` are
        masked out.  Default ``False`` (appropriate for RSVD); set
        ``True`` for GRM construction.

    Returns
    -------
    mask : ndarray of bool, shape (len(variant_ids),)
        ``True`` for variants that should be **retained**.
    """
    ids = np.asarray(variant_ids, dtype=str)
    n = len(ids)

    # --- Fallback: no upstream QC data provided --------------------------
    if qc_data is None:
        warnings.warn(
            "No upstream variant QC data provided; returning all-True mask.  "
            "Provide a collated_variant_qc.tsv for ancestry-informed "
            "marker filtering.",
            stacklevel=2,
        )
        return np.ones(n, dtype=bool)

    # --- Build mask in variant_ids order ---------------------------------
    mask = np.ones(n, dtype=bool)
    missing_ids: list[str] = []

    for i, vid in enumerate(ids):
        rec = qc_data.get(str(vid))
        if rec is None:
            missing_ids.append(vid)
            # Treat missing QC data as failing — conservative default.
            mask[i] = False
            continue
        if require_call_rate and not rec.call_rate_pass:
            mask[i] = False
        if require_hwe and not rec.hwe_pass:
            mask[i] = False
        if require_maf and not rec.maf_pass:
            mask[i] = False

    if missing_ids:
        n_miss = len(missing_ids)
        sample = missing_ids[:5]
        warnings.warn(
            f"{n_miss} variant(s) in the input have no matching QC record "
            f"and will be excluded.  Examples: {sample}",
            stacklevel=2,
        )

    # Check for extra QC records not present in variant_ids.
    id_set = set(ids)
    extra = set(qc_data.keys()) - id_set
    if extra:
        n_extra = len(extra)
        sample_extra = sorted(extra)[:5]
        logger.info(
            "%d QC record(s) have no matching input variant (ignored).  "
            "Examples: %s",
            n_extra,
            sample_extra,
        )

    return mask
