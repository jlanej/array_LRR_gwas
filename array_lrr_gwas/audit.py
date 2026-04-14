"""Centralized audit logger for pipeline filtering decisions.

Provides a structured, extensible mechanism for recording which markers
and samples are included or excluded at each pipeline stage, along with
the reason for each decision.  The audit trail enables full provenance
tracking and reproducibility.

Usage
-----
::

    from array_lrr_gwas.audit import AuditLogger

    audit = AuditLogger()
    audit.record(
        stage="correction_marker_qc",
        id_type="marker",
        included=["chr1:100:A:T", "chr1:200:G:C"],
        excluded={"chr1:300:T:A": "failed_call_rate",
                  "chr1:400:C:G": "failed_hwe"},
    )
    audit.write_tsv("/output/audit_trail.tsv")

Each record captures:

* ``stage`` — pipeline stage (e.g. ``correction_marker_qc``,
  ``grm_variant_qc``, ``association_marker_exclusion``,
  ``sample_qc``)
* ``id_type`` — ``"marker"`` or ``"sample"``
* ``included`` — list of IDs that passed the stage
* ``excluded`` — dict mapping excluded ID → reason string

The audit log is append-only and can be written as a structured TSV
at any point for human readability and downstream parsability.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """A single audit entry for one filtering stage."""

    stage: str
    id_type: str  # "marker" or "sample"
    total_input: int
    total_included: int
    total_excluded: int
    excluded_reasons: dict[str, list[str]] = field(default_factory=dict)
    """Mapping from reason → list of excluded IDs."""


class AuditLogger:
    """Centralized, append-only audit logger for pipeline filtering.

    Collects :class:`AuditRecord` instances across pipeline stages and
    provides write methods for structured output (TSV, JSON).
    """

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        stage: str,
        id_type: str,
        included: Sequence[str],
        excluded: dict[str, str] | None = None,
    ) -> AuditRecord:
        """Record a filtering decision.

        Parameters
        ----------
        stage : str
            Pipeline stage name (e.g. ``"correction_marker_qc"``).
        id_type : str
            ``"marker"`` or ``"sample"``.
        included : sequence of str
            IDs that passed this stage.
        excluded : dict or None
            Mapping of excluded ID → reason string.  ``None`` treated
            as an empty dict.

        Returns
        -------
        AuditRecord
            The recorded entry.
        """
        excluded = excluded or {}

        # Group excluded IDs by reason
        reasons: dict[str, list[str]] = {}
        for eid, reason in excluded.items():
            reasons.setdefault(reason, []).append(eid)

        total_input = len(included) + len(excluded)
        rec = AuditRecord(
            stage=stage,
            id_type=id_type,
            total_input=total_input,
            total_included=len(included),
            total_excluded=len(excluded),
            excluded_reasons=reasons,
        )
        self._records.append(rec)

        logger.info(
            "Audit [%s] %s: %d / %d included, %d excluded",
            stage, id_type, rec.total_included, rec.total_input,
            rec.total_excluded,
        )
        for reason, ids in reasons.items():
            logger.info(
                "  Reason '%s': %d %s(s)",
                reason, len(ids), id_type,
            )

        return rec

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[AuditRecord]:
        """Return a copy of all audit records."""
        return list(self._records)

    def summary(self) -> list[dict[str, object]]:
        """Return a list of summary dicts (one per record)."""
        out = []
        for r in self._records:
            out.append({
                "stage": r.stage,
                "id_type": r.id_type,
                "total_input": r.total_input,
                "total_included": r.total_included,
                "total_excluded": r.total_excluded,
                "excluded_reason_counts": {
                    reason: len(ids)
                    for reason, ids in r.excluded_reasons.items()
                },
            })
        return out

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_tsv(self, path: str | Path) -> Path:
        """Write a detailed per-ID audit trail as a TSV file.

        Each row represents either a per-stage summary (``status=summary``)
        or a single excluded ID (``status=excluded``):

            stage | id_type | id | status | reason

        Summary rows have ``id`` set to the count of included IDs and
        ``reason`` empty, providing a record of how many IDs passed
        each stage without enumerating them individually.

        Parameters
        ----------
        path : str or Path
            Output file path.

        Returns
        -------
        Path
            The written file path.
        """
        path = Path(path)
        fieldnames = ["stage", "id_type", "id", "status", "reason"]

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for rec in self._records:
                # Write a per-stage summary row with included count
                writer.writerow({
                    "stage": rec.stage,
                    "id_type": rec.id_type,
                    "id": f"n={rec.total_included}",
                    "status": "included_summary",
                    "reason": "",
                })
                # Write excluded IDs with reasons
                for reason, ids in rec.excluded_reasons.items():
                    for eid in ids:
                        writer.writerow({
                            "stage": rec.stage,
                            "id_type": rec.id_type,
                            "id": eid,
                            "status": "excluded",
                            "reason": reason,
                        })

        logger.info("Wrote audit trail TSV: %s (%d records)", path, len(self._records))
        return path

    def write_json(self, path: str | Path) -> Path:
        """Write a JSON summary of all audit records.

        Parameters
        ----------
        path : str or Path
            Output file path.

        Returns
        -------
        Path
            The written file path.
        """
        path = Path(path)
        data = {
            "audit_records": [],
        }
        for rec in self._records:
            entry = {
                "stage": rec.stage,
                "id_type": rec.id_type,
                "total_input": rec.total_input,
                "total_included": rec.total_included,
                "total_excluded": rec.total_excluded,
                "excluded_reasons": {
                    reason: {"count": len(ids), "ids": ids}
                    for reason, ids in rec.excluded_reasons.items()
                },
            }
            data["audit_records"].append(entry)

        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

        logger.info("Wrote audit trail JSON: %s (%d records)", path, len(self._records))
        return path

    def write_summary_tsv(self, path: str | Path) -> Path:
        """Write a stage-level summary TSV (one row per stage).

        Columns: stage, id_type, total_input, total_included,
        total_excluded, included_fraction, excluded_fraction,
        excluded_reason_counts

        Parameters
        ----------
        path : str or Path
            Output file path.

        Returns
        -------
        Path
            The written file path.
        """
        path = Path(path)
        fieldnames = [
            "stage", "id_type", "total_input", "total_included",
            "total_excluded", "included_fraction", "excluded_fraction",
            "excluded_reason_counts",
        ]

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for rec in self._records:
                reason_counts = {
                    reason: len(ids)
                    for reason, ids in rec.excluded_reasons.items()
                }
                total = rec.total_input
                inc_frac = (
                    f"{rec.total_included / total:.4f}"
                    if total > 0 else "NA"
                )
                exc_frac = (
                    f"{rec.total_excluded / total:.4f}"
                    if total > 0 else "NA"
                )
                writer.writerow({
                    "stage": rec.stage,
                    "id_type": rec.id_type,
                    "total_input": rec.total_input,
                    "total_included": rec.total_included,
                    "total_excluded": rec.total_excluded,
                    "included_fraction": inc_frac,
                    "excluded_fraction": exc_frac,
                    "excluded_reason_counts": json.dumps(reason_counts),
                })

        logger.info(
            "Wrote audit summary TSV: %s (%d stages)", path, len(self._records),
        )
        return path
