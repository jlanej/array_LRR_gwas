"""Parse upstream compiled sample sheets for pre-computed PCs and covariates.

The ``compiled_sample_sheet.tsv`` produced by the
``jlanej/illumina_idat_processing`` pipeline contains per-sample
metadata including pre-computed global ancestry principal components
(e.g. ``PC1`` through ``PC20``), ancestry-stratified PCs
(e.g. ``EUR_PC1``), predicted sex, and other QC fields.

This module provides utilities to ingest that TSV and extract the
columns needed as fixed-effect covariates for the LMM association,
as well as to classify samples as high-quality (HQ) based on the
``call_rate`` and ``lrr_sd`` columns present in the sample sheet.

Association-stage exclusions
----------------------------
:func:`classify_samples_for_association` extends the basic HQ
classification with additional exclusion criteria recommended for
GWAS best practice:

* **Pre-computed exclusions** — ``pre_pca_excluded``,
  ``excluded_relatedness``, ``excluded_het_outlier`` columns from the
  compiled sample sheet.  These reflect upstream QC decisions
  (e.g. UK Biobank / TOPMed conventions).
* **High BAF SD** — BAF standard deviation > threshold (default 0.15)
  indicates potential sample contamination
  (Marees et al. 2018 BMC Genomics).
* **Sex discordance** — ``sex_status == "DISCORDANT"`` flags possible
  sample swaps (Anderson et al. 2010 Nat Protoc).
* **Extreme inbreeding coefficient** — |F| > threshold (default 0.15)
  as a safety net for extreme population structure or sample issues
  (Anderson et al. 2010).

All exclusions are logged with per-category counts for full provenance.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


def _resolve_column(
    fieldnames: list[str],
    requested: str,
) -> str | None:
    """Resolve *requested* column name against *fieldnames*, case-insensitively.

    Returns the **actual** column name present in the header that matches
    *requested* when compared in a case-insensitive manner.  If no match
    is found, returns ``None``.  If an exact (case-sensitive) match exists
    it is preferred; otherwise the first case-insensitive match wins.
    """
    if requested in fieldnames:
        return requested
    lower = requested.lower()
    for col in fieldnames:
        if col.lower() == lower:
            return col
    return None


def read_sample_sheet(
    path: str | Path,
    *,
    sample_id_col: str = "Sample_ID",
    pc_prefix: str = "PC",
    n_pcs: int = 20,
    extra_covariates: list[str] | None = None,
) -> tuple[list[str], NDArray[np.floating], list[str]]:
    """Read a compiled sample sheet and extract covariates.

    Parameters
    ----------
    path : str or Path
        Path to a tab-separated sample sheet.
    sample_id_col : str
        Name of the column containing sample identifiers.
    pc_prefix : str
        Prefix for global-ancestry PC columns (e.g. ``'PC'`` matches
        ``PC1``, ``PC2``, …).  Only columns matching
        ``^{pc_prefix}\\d+$`` are extracted.
    n_pcs : int
        Maximum number of PCs to include (sorted by index).
    extra_covariates : list of str or None
        Additional column names to include (e.g. ``['predicted_sex']``).

    Returns
    -------
    sample_ids : list of str
        Sample identifiers in row order.
    covariates : ndarray, shape (n_samples, n_covariates)
        Covariate matrix.  Columns are PCs (in order) followed by
        any extra covariates.  Non-numeric values are encoded as
        ``np.nan``.
    covariate_names : list of str
        Names of the covariate columns in the returned matrix.
    """
    path = str(path)
    pc_pattern = re.compile(rf"^{re.escape(pc_prefix)}(\d+)$")

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Sample sheet at {path} is empty or has no header")

        # Resolve sample-ID column (case-insensitive)
        resolved_sid_col = _resolve_column(reader.fieldnames, sample_id_col)
        if resolved_sid_col is None:
            resolved_sid_col = sample_id_col  # fall through; rows will yield None

        # Identify PC columns
        pc_cols: list[tuple[int, str]] = []
        for col in reader.fieldnames:
            m = pc_pattern.match(col)
            if m:
                pc_cols.append((int(m.group(1)), col))
        pc_cols.sort()
        pc_cols = pc_cols[:n_pcs]

        extra = list(extra_covariates) if extra_covariates else []
        cov_names = [c for _, c in pc_cols] + extra

        sample_ids: list[str] = []
        rows: list[list[float]] = []

        for row in reader:
            sid = row.get(resolved_sid_col)
            if sid is None or sid == "":
                continue
            sample_ids.append(sid)

            vals: list[float] = []
            for name in cov_names:
                raw = row.get(name, "")
                try:
                    vals.append(float(raw))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            rows.append(vals)

    if not rows:
        covariates = np.empty((0, len(cov_names)), dtype=np.float64)
    else:
        covariates = np.array(rows, dtype=np.float64)

    return sample_ids, covariates, cov_names


def align_samples(
    target_samples: list[str],
    sheet_samples: list[str],
    covariates: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Align sample-sheet covariates to match a target sample order.

    Parameters
    ----------
    target_samples : list of str
        Desired sample ordering (e.g. from the BCF).
    sheet_samples : list of str
        Sample ordering in the sample sheet.
    covariates : ndarray, shape (n_sheet_samples, n_covariates)
        Covariate matrix from the sample sheet.

    Returns
    -------
    aligned : ndarray, shape (len(target_samples), n_covariates)
        Covariates reordered to match *target_samples*.  Samples
        missing from the sheet are filled with ``np.nan``.
    """
    sheet_idx = {s: i for i, s in enumerate(sheet_samples)}
    n_cov = covariates.shape[1] if covariates.ndim == 2 else 0
    aligned = np.full((len(target_samples), n_cov), np.nan, dtype=np.float64)

    for j, sid in enumerate(target_samples):
        if sid in sheet_idx:
            aligned[j] = covariates[sheet_idx[sid]]

    return aligned


def read_all_raw_rows(
    path: str | Path,
    *,
    sample_id_col: str = "sample_id",
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Read every row of a compiled sample sheet into a raw string dict.

    This is the low-level primitive used by
    :func:`~array_lrr_gwas.interactive_report._parse_sample_sheet_columns`
    to load all columns generically without assuming a fixed schema.

    Supported formats
    -----------------
    * **Plain TSV** — tab-separated, no section headers.  Any file whose
      first non-empty line does *not* start with ``[`` is treated as a
      plain TSV (the standard ``compiled_sample_sheet.tsv`` format).
    * **Illumina multi-section CSV** — files that begin with section
      markers such as ``[Header]``, ``[Manifests]``, ``[Data]``.  Lines
      before the ``[Data]`` marker are skipped; the data section is
      parsed as comma-separated CSV.  This matches the sample-sheet
      format produced by Illumina LIMS / GenomeStudio.

    Parameters
    ----------
    path : str or Path
        Path to a compiled sample sheet (TSV or Illumina CSV).
    sample_id_col : str
        Column name containing sample identifiers (case-insensitive lookup).
        Defaults to ``"sample_id"``.

    Returns
    -------
    other_columns : list of str
        Ordered list of column names present in the sheet, excluding the
        resolved sample-ID column.
    raw : dict mapping sample ID → dict of {column_name: raw_string_value}
        Row data for every sample found in the sheet.  Values are raw
        (unparsed) strings; missing values are empty strings.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        all_lines = fh.readlines()

    if not all_lines:
        return [], {}

    # Detect Illumina multi-section format: first non-empty line starts with '['.
    first_nonempty = next((ln for ln in all_lines if ln.strip()), "")
    if first_nonempty.strip().startswith("["):
        # Find the [Data] section (case-insensitive).
        data_start: int | None = None
        for idx, line in enumerate(all_lines):
            if line.strip().lower() == "[data]":
                data_start = idx + 1  # CSV header is on the very next line
                break
        if data_start is None or data_start >= len(all_lines):
            return [], {}
        data_lines = all_lines[data_start:]
        delimiter = ","
    else:
        data_lines = all_lines
        delimiter = "\t"

    reader = csv.DictReader(io.StringIO("".join(data_lines)), delimiter=delimiter)
    fieldnames: list[str] = list(reader.fieldnames or [])
    if not fieldnames:
        return [], {}

    # Use the shared resolver for case-insensitive column lookup.
    sid_col = _resolve_column(fieldnames, sample_id_col)
    if sid_col is None:
        sid_col = fieldnames[0]

    other_columns = [c for c in fieldnames if c != sid_col]

    raw: dict[str, dict[str, str]] = {}
    for row in reader:
        sid = row.get(sid_col, "").strip()
        if sid:
            raw[sid] = {c: (row.get(c) or "") for c in other_columns}

    return other_columns, raw


def classify_samples_from_sheet(
    path: str | Path,
    *,
    max_lrr_sd: float = 0.35,
    min_call_rate: float = 0.97,
    sample_id_col: str = "Sample_ID",
    call_rate_col: str = "call_rate",
    lrr_sd_col: str = "lrr_sd",
) -> set[str]:
    """Derive a set of high-quality sample IDs from a compiled sample sheet.

    Applies the same QC criteria used by
    :func:`~array_lrr_gwas.correction.classify_samples`:

    * ``call_rate >= min_call_rate``
    * ``lrr_sd <= max_lrr_sd``

    Parameters
    ----------
    path : str or Path
        Path to the tab-separated compiled sample sheet.
    max_lrr_sd : float
        Maximum per-sample LRR standard deviation to be considered HQ.
    min_call_rate : float
        Minimum per-sample genotype call rate to be considered HQ.
    sample_id_col : str
        Name of the column containing sample identifiers.
    call_rate_col : str
        Name of the column containing per-sample call rates.
    lrr_sd_col : str
        Name of the column containing per-sample LRR standard deviations.

    Returns
    -------
    hq_ids : set of str
        Sample IDs that pass both QC thresholds.

    Raises
    ------
    ValueError
        If the sample sheet is missing required columns.
    """
    path = str(path)

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Sample sheet at {path} is empty or has no header")

        # Resolve column names case-insensitively
        resolved_sid = _resolve_column(reader.fieldnames, sample_id_col) or sample_id_col
        resolved_cr = _resolve_column(reader.fieldnames, call_rate_col) or call_rate_col
        resolved_sd = _resolve_column(reader.fieldnames, lrr_sd_col) or lrr_sd_col

        required = {resolved_sid, resolved_cr, resolved_sd}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Sample sheet is missing required columns for HQ "
                f"classification: {sorted(missing)}"
            )

        hq_ids: set[str] = set()
        for row in reader:
            sid = row.get(resolved_sid)
            if sid is None or sid == "":
                continue
            try:
                cr = float(row.get(resolved_cr, ""))
            except (TypeError, ValueError):
                continue
            try:
                sd = float(row.get(resolved_sd, ""))
            except (TypeError, ValueError):
                continue
            if cr >= min_call_rate and sd <= max_lrr_sd:
                hq_ids.add(sid)

    return hq_ids


# ---------------------------------------------------------------------------
# Association-stage sample exclusion
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class ExclusionResult(NamedTuple):
    """Result of association-stage sample exclusion.

    Attributes
    ----------
    hq_ids : set of str
        Sample IDs passing all enabled exclusion criteria.
    counts : dict of str to int
        Number of samples excluded by each category.  Keys include
        ``"low_call_rate"``, ``"high_lrr_sd"``, ``"pre_pca_excluded"``,
        ``"excluded_relatedness"``, ``"excluded_het_outlier"``,
        ``"high_baf_sd"``, ``"sex_discordant"``, ``"extreme_inbreeding_f"``,
        and ``"total_excluded"``.
    total : int
        Total number of samples in the sheet.
    excluded_reasons : dict of str to list of str
        Per-sample exclusion reasons.  Maps each excluded sample ID to
        a list of reason strings (e.g. ``["low_call_rate", "high_lrr_sd"]``).
    """

    hq_ids: set[str]
    counts: dict[str, int]
    total: int
    excluded_reasons: dict[str, list[str]]


def _parse_bool_field(val: str | None) -> bool | None:
    """Parse a boolean-ish sample-sheet value.

    Returns ``True`` for ``'true'``, ``'1'``, ``'yes'`` (case-insensitive),
    ``False`` for ``'false'``, ``'0'``, ``'no'``, and ``None`` otherwise.
    """
    if val is None:
        return None
    txt = val.strip().lower()
    if txt in {"true", "1", "yes"}:
        return True
    if txt in {"false", "0", "no"}:
        return False
    return None


def classify_samples_for_association(
    path: str | Path,
    *,
    max_lrr_sd: float = 0.35,
    min_call_rate: float = 0.97,
    honor_precomputed: bool = True,
    exclude_baf_sd: bool = True,
    max_baf_sd: float = 0.15,
    exclude_sex_discordant: bool = True,
    exclude_extreme_inbreeding: bool = True,
    max_abs_inbreeding_f: float = 0.15,
    sample_id_col: str = "Sample_ID",
    call_rate_col: str = "call_rate",
    lrr_sd_col: str = "lrr_sd",
    baf_sd_col: str = "baf_sd",
    sex_status_col: str = "sex_status",
    inbreeding_f_col: str = "inbreeding_F",
    pre_pca_excluded_col: str = "pre_pca_excluded",
    excluded_relatedness_col: str = "excluded_relatedness",
    excluded_het_outlier_col: str = "excluded_het_outlier",
) -> ExclusionResult:
    """Derive high-quality sample IDs for association analysis.

    Applies the core HQ criteria (call rate + LRR SD) used by
    :func:`classify_samples_from_sheet`, plus additional exclusion
    criteria recommended for GWAS best practice.

    **Default exclusions (always applied):**

    * ``call_rate >= min_call_rate`` (default 0.97)
    * ``lrr_sd <= max_lrr_sd`` (default 0.35)

    **Pre-computed upstream exclusions** (``honor_precomputed=True``):

    * ``pre_pca_excluded`` — sample excluded before ancestry PCA
      (e.g. failed upstream genotype QC).
    * ``excluded_relatedness`` — removed as part of a related pair
      (typically up to 2nd degree, kinship > 0.0884; UK Biobank and
      TOPMed convention).  While the GRM/LMM can correct for moderate
      kinship, removing close relatives reduces bias in effect estimates
      and avoids overcounting families.
    * ``excluded_het_outlier`` — extreme heterozygosity outlier,
      suggesting potential DNA contamination or sample mix-up.

    **Optional exclusions** (enabled by default per user request):

    * ``exclude_baf_sd`` — high BAF standard deviation (> ``max_baf_sd``,
      default 0.15) indicates potential sample contamination
      (Marees et al. 2018).
    * ``exclude_sex_discordant`` — ``sex_status == "DISCORDANT"``
      flags possible sample swaps (Anderson et al. 2010).
    * ``exclude_extreme_inbreeding`` — |inbreeding_F| > threshold
      (default 0.15) as a safety net for extreme population structure
      or sample issues (Anderson et al. 2010).

    Missing columns for optional criteria are silently skipped (with a
    logged info message).  Missing or non-parseable values for a sample
    cause that sample to be treated as *not excluded* for that criterion.

    Parameters
    ----------
    path : str or Path
        Path to the tab-separated compiled sample sheet.
    max_lrr_sd : float
        Maximum per-sample LRR standard deviation.
    min_call_rate : float
        Minimum per-sample genotype call rate.
    honor_precomputed : bool
        If True (default), exclude samples flagged by ``pre_pca_excluded``,
        ``excluded_relatedness``, or ``excluded_het_outlier``.
    exclude_baf_sd : bool
        If True (default), exclude samples with ``baf_sd > max_baf_sd``.
    max_baf_sd : float
        BAF SD threshold (default 0.15).
    exclude_sex_discordant : bool
        If True (default), exclude samples with
        ``sex_status == "DISCORDANT"``.
    exclude_extreme_inbreeding : bool
        If True (default), exclude samples with
        ``|inbreeding_F| > max_abs_inbreeding_f``.
    max_abs_inbreeding_f : float
        Inbreeding coefficient threshold (default 0.15).
    sample_id_col, call_rate_col, lrr_sd_col, baf_sd_col,
    sex_status_col, inbreeding_f_col, pre_pca_excluded_col,
    excluded_relatedness_col, excluded_het_outlier_col : str
        Column names in the sample sheet.

    Returns
    -------
    ExclusionResult
        Named tuple with ``hq_ids`` (set of str), ``counts`` (dict),
        and ``total`` (int).

    Raises
    ------
    ValueError
        If the sample sheet is missing the required core columns
        (``sample_id_col``, ``call_rate_col``, ``lrr_sd_col``).

    References
    ----------
    * Anderson et al. 2010 *Nat Protoc* 5:1564–1573.
    * Marees et al. 2018 *BMC Genomics* (previously *Int J Methods
      Psychiatr Res*) — GWAS QC tutorial.
    * UK Biobank genotyping QC documentation.
    * TOPMed analysis best practices.
    """
    path = str(path)

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Sample sheet at {path} is empty or has no header")

        # Resolve column names case-insensitively
        resolved_sid = _resolve_column(reader.fieldnames, sample_id_col) or sample_id_col
        resolved_cr = _resolve_column(reader.fieldnames, call_rate_col) or call_rate_col
        resolved_sd = _resolve_column(reader.fieldnames, lrr_sd_col) or lrr_sd_col

        # Core columns are always required
        required = {resolved_sid, resolved_cr, resolved_sd}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Sample sheet is missing required columns for HQ "
                f"classification: {sorted(missing)}"
            )

        available_cols = set(reader.fieldnames)

        # Resolve optional column names case-insensitively
        resolved_pre_pca = _resolve_column(reader.fieldnames, pre_pca_excluded_col)
        resolved_relatedness = _resolve_column(reader.fieldnames, excluded_relatedness_col)
        resolved_het_outlier = _resolve_column(reader.fieldnames, excluded_het_outlier_col)
        resolved_baf_sd = _resolve_column(reader.fieldnames, baf_sd_col)
        resolved_sex_status = _resolve_column(reader.fieldnames, sex_status_col)
        resolved_inbreeding = _resolve_column(reader.fieldnames, inbreeding_f_col)

        # Check optional column availability
        has_pre_pca = resolved_pre_pca is not None
        has_relatedness = resolved_relatedness is not None
        has_het_outlier = resolved_het_outlier is not None
        has_baf_sd = resolved_baf_sd is not None
        has_sex_status = resolved_sex_status is not None
        has_inbreeding = resolved_inbreeding is not None

        if honor_precomputed:
            if not has_pre_pca:
                logger.info(
                    "Column '%s' not found; skipping pre-PCA exclusion",
                    pre_pca_excluded_col,
                )
            if not has_relatedness:
                logger.info(
                    "Column '%s' not found; skipping relatedness exclusion",
                    excluded_relatedness_col,
                )
            if not has_het_outlier:
                logger.info(
                    "Column '%s' not found; skipping het-outlier exclusion",
                    excluded_het_outlier_col,
                )
        if exclude_baf_sd and not has_baf_sd:
            logger.info(
                "Column '%s' not found; skipping BAF SD exclusion",
                baf_sd_col,
            )
        if exclude_sex_discordant and not has_sex_status:
            logger.info(
                "Column '%s' not found; skipping sex-discordant exclusion",
                sex_status_col,
            )
        if exclude_extreme_inbreeding and not has_inbreeding:
            logger.info(
                "Column '%s' not found; skipping inbreeding-F exclusion",
                inbreeding_f_col,
            )

        counts: dict[str, int] = {
            "low_call_rate": 0,
            "high_lrr_sd": 0,
            "pre_pca_excluded": 0,
            "excluded_relatedness": 0,
            "excluded_het_outlier": 0,
            "high_baf_sd": 0,
            "sex_discordant": 0,
            "extreme_inbreeding_f": 0,
            "total_excluded": 0,
        }
        total = 0
        hq_ids: set[str] = set()
        per_sample_reasons: dict[str, list[str]] = {}

        for row in reader:
            sid = row.get(resolved_sid)
            if sid is None or sid == "":
                continue
            total += 1
            excluded = False
            reasons: list[str] = []

            # Core QC: call rate
            try:
                cr = float(row.get(resolved_cr, ""))
            except (TypeError, ValueError):
                cr = None
            if cr is None or cr < min_call_rate:
                counts["low_call_rate"] += 1
                excluded = True
                reasons.append("low_call_rate")

            # Core QC: LRR SD
            try:
                sd = float(row.get(resolved_sd, ""))
            except (TypeError, ValueError):
                sd = None
            if sd is None or sd > max_lrr_sd:
                counts["high_lrr_sd"] += 1
                excluded = True
                reasons.append("high_lrr_sd")

            # Pre-computed exclusions
            if honor_precomputed:
                if has_pre_pca and _parse_bool_field(row.get(resolved_pre_pca)) is True:
                    counts["pre_pca_excluded"] += 1
                    excluded = True
                    reasons.append("pre_pca_excluded")
                if has_relatedness and _parse_bool_field(row.get(resolved_relatedness)) is True:
                    counts["excluded_relatedness"] += 1
                    excluded = True
                    reasons.append("excluded_relatedness")
                if has_het_outlier and _parse_bool_field(row.get(resolved_het_outlier)) is True:
                    counts["excluded_het_outlier"] += 1
                    excluded = True
                    reasons.append("excluded_het_outlier")

            # BAF SD exclusion (contamination proxy)
            if exclude_baf_sd and has_baf_sd:
                try:
                    baf = float(row.get(resolved_baf_sd, ""))
                except (TypeError, ValueError):
                    baf = None
                if baf is not None and baf > max_baf_sd:
                    counts["high_baf_sd"] += 1
                    excluded = True
                    reasons.append("high_baf_sd")

            # Sex discordance exclusion
            if exclude_sex_discordant and has_sex_status:
                sex_val = (row.get(resolved_sex_status) or "").strip().upper()
                if sex_val == "DISCORDANT":
                    counts["sex_discordant"] += 1
                    excluded = True
                    reasons.append("sex_discordant")

            # Extreme inbreeding F exclusion
            if exclude_extreme_inbreeding and has_inbreeding:
                try:
                    f_val = float(row.get(resolved_inbreeding, ""))
                except (TypeError, ValueError):
                    f_val = None
                if f_val is not None and abs(f_val) > max_abs_inbreeding_f:
                    counts["extreme_inbreeding_f"] += 1
                    excluded = True
                    reasons.append("extreme_inbreeding_f")

            if excluded:
                counts["total_excluded"] += 1
                per_sample_reasons[sid] = reasons
            else:
                hq_ids.add(sid)

    return ExclusionResult(
        hq_ids=hq_ids, counts=counts, total=total,
        excluded_reasons=per_sample_reasons,
    )
