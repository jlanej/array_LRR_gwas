"""YAML-based QC configuration for sample and marker filtering.

Provides best-practice defaults aligned with the upstream
``jlanej/illumina_idat_processing`` pipeline and GWAS QC literature
(Anderson et al. 2010; Marees et al. 2018), with full override capacity
via a YAML configuration file.

Best-Practice Default Thresholds
---------------------------------

**Sample QC** (HQ / LQ classification):

* ``max_lrr_sd``: **0.35** — Maximum per-sample LRR standard deviation.
  Samples exceeding this are classified as low-quality (LQ).
  Matches upstream ``filter_qc_samples.py --max-lrr-sd 0.35``.
* ``min_call_rate``: **0.97** — Minimum per-sample genotype call rate
  (autosomes).  Matches upstream ``filter_qc_samples.py --min-call-rate
  0.97`` and standard GWAS QC guidance (≥ 0.95–0.98).

**Marker QC** (batch-correction subsetting):

* ``min_call_rate``: **0.95** — Minimum per-marker call rate.  Markers
  below this threshold are excluded from the decomposition.
* ``min_var``: **0.001** — Minimum per-marker LRR variance; removes
  uninformative (near-constant) markers from the decomposition.
* ``max_var``: **None** — Maximum per-marker LRR variance.  Disabled by
  default; set to remove artefactual high-variance outliers if needed.

**Correction parameters**:

* ``k``: **None** — Number of batch-effect components to remove.
  ``None`` triggers automatic selection via the Marchenko–Pastur heuristic.
* ``backend``: **"rsvd"** — Decomposition backend.  Options: ``rsvd``
  (scikit-learn randomised SVD) or ``fbpca`` (Facebook PCA, optional).
* ``no_complexity_filter``: **False** — When True, skips the default
  genomic-complexity region exclusion (centromeres, segdups).

Example YAML
-------------
.. code-block:: yaml

    # Override any subset of defaults.  Omitted keys keep their defaults.
    sample_qc:
      max_lrr_sd: 0.30        # stricter noise threshold
      min_call_rate: 0.98      # stricter call-rate threshold

    marker_qc:
      min_call_rate: 0.98
      min_var: 0.002
      max_var: 5.0             # exclude extreme-variance markers

    correction:
      k: 5                     # fix number of batch components
      backend: rsvd
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Best-practice default configuration
# ---------------------------------------------------------------------------
# Each value is explicitly documented above and in docs/upstream_qc_formats.md.
# Sources: upstream jlanej/illumina_idat_processing filter_qc_samples.py,
# Anderson et al. 2010 Nat Protoc, Marees et al. 2018 IJMPR.

_DEFAULTS: dict[str, Any] = {
    "sample_qc": {
        # Max per-sample LRR SD for HQ classification.
        # Upstream default: 0.35  (filter_qc_samples.py)
        "max_lrr_sd": 0.35,
        # Min per-sample call rate for HQ classification.
        # Upstream default: 0.97  (filter_qc_samples.py)
        "min_call_rate": 0.97,
    },
    "marker_qc": {
        # Min per-marker call rate for batch-correction subsetting.
        # A moderately inclusive threshold retains more markers for SVD.
        "min_call_rate": 0.95,
        # Min per-marker LRR variance; removes near-constant markers.
        "min_var": 0.001,
        # Max per-marker LRR variance; None = no upper limit.
        "max_var": None,
    },
    "correction": {
        # Number of batch PCs to remove; None = auto (Marchenko-Pastur).
        "k": None,
        # Decomposition backend: "rsvd" or "fbpca".
        "backend": "rsvd",
        # Skip genomic-complexity region exclusion if True.
        "no_complexity_filter": False,
    },
}

_VALID_SECTIONS = frozenset(_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def defaults() -> dict[str, Any]:
    """Return a deep copy of the best-practice default QC configuration.

    Every key is documented in :mod:`array_lrr_gwas.qc_config` and in
    ``docs/upstream_qc_formats.md``.
    """
    return copy.deepcopy(_DEFAULTS)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML QC configuration file and merge with defaults.

    Only recognised top-level sections (``sample_qc``, ``marker_qc``,
    ``correction``) are merged; unknown sections raise :class:`ValueError`.
    Keys that are absent in the YAML file retain their best-practice
    defaults, so users only need to specify the values they wish to
    override.

    Parameters
    ----------
    path : str or Path
        Path to a YAML file.

    Returns
    -------
    dict
        Merged configuration (defaults + overrides).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML file contains unrecognised top-level keys or
        malformed sections.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        user: dict[str, Any] = yaml.safe_load(fh) or {}

    if not isinstance(user, dict):
        raise ValueError(
            f"Expected a YAML mapping at the top level, got {type(user).__name__}"
        )

    unknown = set(user.keys()) - _VALID_SECTIONS
    if unknown:
        raise ValueError(
            f"Unrecognised config sections: {sorted(unknown)}. "
            f"Valid sections: {sorted(_VALID_SECTIONS)}"
        )

    merged = defaults()
    for section in _VALID_SECTIONS:
        if section in user:
            if not isinstance(user[section], dict):
                raise ValueError(
                    f"Section '{section}' must be a mapping, "
                    f"got {type(user[section]).__name__}"
                )
            merged[section].update(user[section])

    return merged


def apply_to_correct_args(
    cfg: dict[str, Any],
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a merged QC config into keyword arguments for
    :func:`~array_lrr_gwas.correction.correct_lrr`.

    **Precedence**: CLI flags > YAML config > built-in defaults.

    Parameters
    ----------
    cfg : dict
        Merged configuration from :func:`load_config` or :func:`defaults`.
    cli_overrides : dict or None
        Explicit CLI overrides (only non-``None`` values are applied).

    Returns
    -------
    dict
        Keyword arguments suitable for ``correct_lrr()``.
    """
    args: dict[str, Any] = {
        "max_lrr_sd": cfg["sample_qc"]["max_lrr_sd"],
        "min_sample_call_rate": cfg["sample_qc"]["min_call_rate"],
        "min_marker_call_rate": cfg["marker_qc"]["min_call_rate"],
        "min_var": cfg["marker_qc"]["min_var"],
        "max_var": cfg["marker_qc"]["max_var"],
        "k": cfg["correction"]["k"],
        "backend": cfg["correction"]["backend"],
    }

    if cli_overrides:
        for key, val in cli_overrides.items():
            if val is not None and key in args:
                args[key] = val

    return args
