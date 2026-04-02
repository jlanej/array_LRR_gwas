"""Pipeline audit trail for reproducibility and provenance tracking.

Writes a machine-readable JSON manifest alongside each pipeline output,
recording every input file, parameter, and per-step filtering decision
so that users have a complete record of what was included or excluded
and why.
"""

from __future__ import annotations

import json
import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy / pathlib types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


class AuditTrail:
    """Accumulates pipeline-step metadata for provenance tracking.

    Usage::

        audit = AuditTrail("correct")
        audit.set_input(file="input.bcf", n_variants=96869, n_samples=8)
        audit.set_parameters(max_lrr_sd=0.35, ...)
        audit.add_step("variant_qc_filter", n_pass=80000, n_excluded=16869)
        audit.set_output(file="corrected.bcf")
        audit.write("corrected.bcf.audit.json")
    """

    def __init__(self, command: str) -> None:
        self._data: dict[str, Any] = {
            "command": command,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "inputs": {},
            "parameters": {},
            "steps": [],
            "output": {},
        }

    # -- Setters ------------------------------------------------------------

    def set_input(self, **kwargs: Any) -> None:
        """Record input-file metadata."""
        self._data["inputs"].update(kwargs)

    def set_parameters(self, **kwargs: Any) -> None:
        """Record analysis parameters / thresholds."""
        self._data["parameters"].update(kwargs)

    def add_step(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Append a pipeline step with its metadata.

        Returns the step dict for optional further mutation.
        """
        step: dict[str, Any] = {"step": name}
        step.update(kwargs)
        self._data["steps"].append(step)
        return step

    def set_output(self, **kwargs: Any) -> None:
        """Record output-file metadata."""
        self._data["output"].update(kwargs)

    # -- Accessors ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the complete audit dict (for testing / inspection)."""
        return self._data

    # -- I/O ----------------------------------------------------------------

    def write(self, path: str | Path) -> Path:
        """Serialise the audit trail to a JSON file.

        Returns the path written to.
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, default=_json_default)
            fh.write("\n")
        logger.info("Wrote audit trail: %s", path)
        return path


def audit_path_for(output: Path) -> Path:
    """Derive the audit JSON path from a pipeline output path.

    ``results.tsv`` → ``results.tsv.audit.json``
    ``corrected.bcf`` → ``corrected.bcf.audit.json``
    """
    return Path(str(output) + ".audit.json")
