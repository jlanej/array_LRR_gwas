#!/usr/bin/env bash
# test_run_correction.sh — Run array-lrr-gwas correct on the 100-sample test BCF.
#
# Input
# -----
#   tests/data/stage2_reclustered.100.subsample.subset.bcf
#     100-sample, 12 109-variant subset of an Illumina array BCF.
#     Genome build: T2T-CHM13 (CHM13v2.0).
#     FORMAT fields: GT, BAF, LRR, IGC, GQ.
#
# Output (written to output/correction/)
# ----------------------------------------
#   corrected.bcf              — batch-effect-corrected LRR values
#   corrected.bcf.csi          — index
#   corrected.bcf.svd.sample_pcs.tsv    — per-sample batch PCs
#   corrected.bcf.svd.singular_values.tsv
#   corrected.bcf.svd.loadings.tsv      — per-marker PC loadings
#                                          (--write-loadings)
#
# Usage
# -----
#   bash scripts/run_correction.sh
#
#   # Override output directory:
#   OUT_DIR=/my/output bash scripts/run_correction.sh
#
#   # Override number of correction components:
#   K=10 bash scripts/run_correction.sh
#
# Notes
# -----
# * --build T2T-CHM13 is specified explicitly; the pipeline auto-detects
#   from the chr1 contig length (248 387 328) but being explicit avoids
#   any ambiguity and skips the pysam detection pass.
#
# * --write-loadings is included so the per-marker PC loadings are
#   available for downstream inspection and sanity-checking.
#
# * --k is left unset by default so the Marchenko–Pastur heuristic
#   selects the number of batch components automatically.  Set K=<int>
#   in your environment to override.
#
# * No --variant-qc is specified for this test run.  The pipeline will
#   log a warning; this is expected when running on the test subset
#   without a collated_variant_qc.tsv file.
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths relative to the repo root
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Auto-activate the project virtual environment if not already active
# ---------------------------------------------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    VENV_ACTIVATE="${REPO_ROOT}/.venv/bin/activate"
    if [[ -f "${VENV_ACTIVATE}" ]]; then
        # shellcheck source=/dev/null
        source "${VENV_ACTIVATE}"
        echo "Activated venv: ${VIRTUAL_ENV}"
    fi
fi

BCF="${REPO_ROOT}/tests/data/stage2_reclustered.100.subsample.subset.bcf"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/output/correction}"
OUT_BCF="${OUT_DIR}/corrected.bcf"

# Optional: set K in your environment to fix the number of components.
# When unset, the pipeline uses the Marchenko–Pastur auto-selection.
K_ARG=""
if [[ -n "${K:-}" ]]; then
    K_ARG="--k ${K}"
fi

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [[ ! -f "${BCF}" ]]; then
    echo "ERROR: Input BCF not found: ${BCF}" >&2
    exit 1
fi

if ! command -v array-lrr-gwas &>/dev/null; then
    echo "ERROR: array-lrr-gwas not found on PATH." >&2
    echo "       Activate the project virtual environment first:" >&2
    echo "         source .venv/bin/activate" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Run correction
# ---------------------------------------------------------------------------
echo "=== array-lrr-gwas correct ==="
echo "  Input  : ${BCF}"
echo "  Output : ${OUT_BCF}"
echo "  Build  : T2T-CHM13"
echo "  k      : ${K:-auto (Marchenko–Pastur)}"
echo ""

# shellcheck disable=SC2086  # K_ARG is intentionally word-split
array-lrr-gwas correct \
    "${BCF}" \
    --build T2T-CHM13 \
    --output "${OUT_BCF}" \
    --svd-output-prefix "${OUT_BCF}.svd" \
    --write-loadings \
    ${K_ARG} \
    --verbose

# Index the corrected BCF
echo ""
echo "Indexing ${OUT_BCF} ..."
bcftools index "${OUT_BCF}"

echo ""
echo "Done.  Outputs written to ${OUT_DIR}/"
ls -lh "${OUT_DIR}/"

