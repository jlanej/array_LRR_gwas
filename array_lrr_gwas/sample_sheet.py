"""Parse upstream compiled sample sheets for pre-computed PCs and covariates.

The ``compiled_sample_sheet.tsv`` produced by the
``jlanej/illumina_idat_processing`` pipeline contains per-sample
metadata including pre-computed global ancestry principal components
(e.g. ``PC1`` through ``PC20``), ancestry-stratified PCs
(e.g. ``EUR_PC1``), predicted sex, and other QC fields.

This module provides utilities to ingest that TSV and extract the
columns needed as fixed-effect covariates for the LMM association.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


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
            sid = row.get(sample_id_col)
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
