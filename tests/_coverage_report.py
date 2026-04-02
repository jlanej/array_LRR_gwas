#!/usr/bin/env python3
"""Comprehensive marker-coverage report between collated_variant_qc.tsv and the BCF."""
import csv
import sys
import collections
import pysam
from pathlib import Path

BCF = Path("tests/data/stage2_reclustered.100.subsample.subset.bcf")
QC  = Path("tests/data/collated_variant_qc.tsv")
OUT = Path("tests/data/marker_coverage_report.md")

# ── 1. Collect BCF variant IDs + positional metadata ─────────────────────────
print("Reading BCF …", flush=True)
bcf_records = {}   # id -> {"chrom", "pos", "ref", "alts"}
bcf_null_id = []   # records with no / dot ID

with pysam.VariantFile(str(BCF)) as vcf:
    for rec in vcf:
        vid = rec.id
        meta = {
            "chrom": rec.chrom,
            "pos":   rec.pos,
            "ref":   rec.ref,
            "alts":  ",".join(rec.alts) if rec.alts else ".",
        }
        if vid and vid != ".":
            bcf_records[vid] = meta
        else:
            bcf_null_id.append(meta)

print(f"  BCF records with ID    : {len(bcf_records):>8,}")
print(f"  BCF records without ID : {len(bcf_null_id):>8,}")

# ── 2. Collect QC variant IDs + key QC columns ───────────────────────────────
print("Reading QC file …", flush=True)
qc_records = {}
KEEP_COLS = ["all_call_rate", "all_maf", "all_hwe_p", "all_ancestries_qc_pass"]

with QC.open(newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    for row in reader:
        vid = row["variant_id"]
        qc_records[vid] = {c: row.get(c, "NA") for c in KEEP_COLS}

print(f"  QC rows                : {len(qc_records):>8,}")

# ── 3. Set arithmetic ─────────────────────────────────────────────────────────
bcf_ids = set(bcf_records)
qc_ids  = set(qc_records)

in_both  = bcf_ids & qc_ids
bcf_only = bcf_ids - qc_ids
qc_only  = qc_ids  - bcf_ids

print(f"\n  Overlap (in both)      : {len(in_both):>8,}")
print(f"  BCF-only (no QC entry) : {len(bcf_only):>8,}")
print(f"  QC-only  (not in BCF)  : {len(qc_only):>8,}")

# ── 4. Characterise BCF-only markers ─────────────────────────────────────────
chrom_counter_bcf = collections.Counter(
    bcf_records[v]["chrom"] for v in bcf_only
)

# ── 5. Characterise QC-only markers ──────────────────────────────────────────
qc_only_pass    = [v for v in qc_only if qc_records[v]["all_ancestries_qc_pass"] == "1"]
qc_only_fail    = [v for v in qc_only if qc_records[v]["all_ancestries_qc_pass"] == "0"]
qc_only_unknown = [v for v in qc_only if qc_records[v]["all_ancestries_qc_pass"] not in ("0", "1")]

# ── 6. Build report ───────────────────────────────────────────────────────────
lines = []
A = lines.append

A("# Marker Coverage Report")
A("")
A("**Files compared**")
A(f"- BCF : `{BCF}`")
A(f"- QC  : `{QC}`")
A("")
A("---")
A("")
A("## 1. Summary")
A("")
A("| Category | Count |")
A("|---|---:|")
A(f"| BCF records (with named ID) | {len(bcf_records):,} |")
A(f"| BCF records *without* a named ID | {len(bcf_null_id):,} |")
A(f"| QC rows | {len(qc_records):,} |")
A(f"| **Markers in both** | **{len(in_both):,}** |")
A(f"| BCF markers absent from QC file | {len(bcf_only):,} |")
A(f"| QC markers absent from BCF | {len(qc_only):,} |")
A("")

# ── Section 2: BCF-only ───────────────────────────────────────────────────────
A("---")
A("")
A(f"## 2. BCF markers absent from QC file  ({len(bcf_only):,} total)")
A("")
if bcf_only:
    A("### 2a. Breakdown by chromosome")
    A("")
    A("| Chromosome | Count |")
    A("|---|---:|")
    for chrom, cnt in sorted(
        chrom_counter_bcf.items(),
        key=lambda x: (x[0].lstrip("chr").zfill(5), x[0]),
    ):
        A(f"| {chrom} | {cnt:,} |")
    A("")
    A("### 2b. Full list (all BCF-only markers)")
    A("")
    A("| variant_id | chrom | pos | ref | alts |")
    A("|---|---|---:|---|---|")
    for vid in sorted(bcf_only):
        m = bcf_records[vid]
        A(f"| `{vid}` | {m['chrom']} | {m['pos']:,} | {m['ref']} | {m['alts']} |")
    A("")
else:
    A("_All BCF markers are present in the QC file._")
    A("")

# ── Section 3: BCF records without an ID ─────────────────────────────────────
A("---")
A("")
A(f"## 3. BCF records without a named ID  ({len(bcf_null_id):,} total)")
A("")
if bcf_null_id:
    A("| chrom | pos | ref | alts |")
    A("|---|---:|---|---|")
    for m in sorted(bcf_null_id, key=lambda x: (x["chrom"], x["pos"])):
        A(f"| {m['chrom']} | {m['pos']:,} | {m['ref']} | {m['alts']} |")
    A("")
else:
    A("_All BCF records have a named ID._")
    A("")

# ── Section 4: QC-only ────────────────────────────────────────────────────────
A("---")
A("")
A(f"## 4. QC markers absent from BCF  ({len(qc_only):,} total)")
A("")
A("These markers exist in the QC file but are not present in the test BCF.  ")
A("This is expected — the BCF is a small 100-sample, 12 109-variant subset of the full array.")
A("")
A("### 4a. QC-pass breakdown")
A("")
A("| QC status | Count |")
A("|---|---:|")
A(f"| `all_ancestries_qc_pass = 1` (would pass) | {len(qc_only_pass):,} |")
A(f"| `all_ancestries_qc_pass = 0` (would fail) | {len(qc_only_fail):,} |")
A(f"| other / missing value | {len(qc_only_unknown):,} |")
A("")

def _sample_table(ids, qc_records, n=20, heading=""):
    rows = []
    if not ids:
        rows.append(f"_No markers._")
        rows.append("")
        return rows
    sample = sorted(ids)[:n]
    rows.append(f"#### {heading} — first {min(n, len(ids)):,} of {len(ids):,}")
    rows.append("")
    rows.append("| variant_id | all_call_rate | all_maf | all_hwe_p | all_ancestries_qc_pass |")
    rows.append("|---|---:|---:|---:|---:|")
    for vid in sample:
        q = qc_records[vid]
        rows.append(
            f"| `{vid}` | {q['all_call_rate']} | {q['all_maf']} "
            f"| {q['all_hwe_p']} | {q['all_ancestries_qc_pass']} |"
        )
    rows.append("")
    return rows

A("### 4b. Sample rows from QC-only markers")
A("")
lines.extend(_sample_table(qc_only_pass,    qc_records, 20, "Would-pass markers (all_ancestries_qc_pass = 1)"))
lines.extend(_sample_table(qc_only_fail,    qc_records, 20, "Would-fail markers (all_ancestries_qc_pass = 0)"))
lines.extend(_sample_table(qc_only_unknown, qc_records, 20, "Unknown / other QC value"))

# ── Section 5: Matched markers ────────────────────────────────────────────────
A("---")
A("")
A(f"## 5. Matched markers  ({len(in_both):,} total)")
A("")
A("These markers appear in both files and will receive QC annotations during `correct` / `associate`.")
A("")
A("| variant_id | chrom | pos | all_call_rate | all_maf | all_hwe_p | all_ancestries_qc_pass |")
A("|---|---|---:|---:|---:|---:|---:|")
for vid in sorted(in_both):
    m = bcf_records[vid]
    q = qc_records[vid]
    A(
        f"| `{vid}` | {m['chrom']} | {m['pos']:,} "
        f"| {q['all_call_rate']} | {q['all_maf']} "
        f"| {q['all_hwe_p']} | {q['all_ancestries_qc_pass']} |"
    )
A("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport written → {OUT}  ({OUT.stat().st_size:,} bytes)")

