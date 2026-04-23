"""Functional tests exercising the pipeline against the real test BCF.

The bundled ``tests/data/test.bcf`` is the output of
``illumina_idat_processing`` and contains 96 869 variants across 8 samples
with FORMAT/LRR and FORMAT/GT fields.  These tests verify that every major
pipeline stage works end-to-end on real data with deterministic mock
phenotypes, mock QC, and mock sample sheets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import (
    BCF_DETECTED_BUILD,
    BCF_N_SAMPLES,
    BCF_N_VARIANTS,
    BCF_SAMPLES,
    STAGE2_N_SAMPLES,
    STAGE2_N_VARIANTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plink2_available() -> bool:
    """Return True if plink2 is found on PATH."""
    import shutil
    return shutil.which("plink2") is not None


def _mock_phenotype(n: int, *, seed: int = 42) -> np.ndarray:
    """Deterministic continuous phenotype."""
    return np.random.default_rng(seed).normal(size=n)


def _mock_binary_phenotype(n: int, *, seed: int = 42) -> np.ndarray:
    """Deterministic binary (0/1) phenotype."""
    return (np.random.default_rng(seed).random(n) > 0.5).astype(float)


def _sanitize_lrr(lrr: np.ndarray) -> np.ndarray:
    """Replace non-finite LRR values (e.g. -inf) with NaN for stable analysis."""
    return np.where(np.isfinite(lrr), lrr, np.nan)


def _write_phenotype_tsv(
    path: Path,
    samples: list[str],
    phenotype: np.ndarray,
) -> None:
    """Write a two-column phenotype file."""
    lines = ["sample_id\tphenotype"]
    for s, y in zip(samples, phenotype):
        lines.append(f"{s}\t{y:.8f}")
    path.write_text("\n".join(lines) + "\n")


def _write_sample_sheet(
    path: Path,
    samples: list[str],
    *,
    n_pcs: int = 3,
    seed: int = 0,
    include_qc: bool = False,
) -> None:
    """Write a mock sample sheet with PCs and optional QC columns."""
    rng = np.random.default_rng(seed)
    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
    header = ["Sample_ID"] + pc_cols
    if include_qc:
        header += ["call_rate", "lrr_sd"]
    lines = ["\t".join(header)]
    for s in samples:
        vals = [f"{v:.6f}" for v in rng.normal(size=n_pcs)]
        row = [s] + vals
        if include_qc:
            row.append(f"{rng.uniform(0.95, 1.0):.4f}")
            row.append(f"{rng.uniform(0.05, 0.30):.4f}")
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")


def _write_variant_qc_tsv(
    path: Path, variants: list[dict], *, fail_first: bool = True
) -> None:
    """Write a mock collated_variant_qc.tsv aligned to *variants*."""
    lines = [
        "variant_id\tall_ancestries_call_rate_pass\t"
        "all_ancestries_hwe_pass\tall_ancestries_maf_pass"
    ]
    for i, v in enumerate(variants):
        vid = v.get("id")
        if not vid:
            alts = ":".join(v.get("alts", ()))
            vid = f"{v['chrom']}:{v['pos']}:{v.get('ref', '')}:{alts}"
        cr = "False" if (fail_first and i == 0) else "True"
        lines.append(f"{vid}\t{cr}\tTrue\tTrue")
    path.write_text("\n".join(lines) + "\n")


# ===================================================================
# 1. BCF I/O
# ===================================================================


class TestBcfIo:
    """Functional tests for reading / writing BCF data."""

    def test_read_lrr_dimensions(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, samples, variants = read_lrr(test_bcf_path)
        assert lrr.shape == (BCF_N_VARIANTS, BCF_N_SAMPLES)
        assert samples == BCF_SAMPLES
        assert len(variants) == BCF_N_VARIANTS

    def test_lrr_contains_finite_values(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, _, _ = read_lrr(test_bcf_path)
        # Most values should be finite (allow small fraction of NaN / inf)
        finite_frac = np.isfinite(lrr).sum() / lrr.size
        assert finite_frac > 0.99

    def test_variant_metadata_complete(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_lrr

        _, _, variants = read_lrr(test_bcf_path)
        for v in variants[:10]:
            assert isinstance(v["chrom"], str)
            assert isinstance(v["pos"], int)
            assert "ref" in v
            assert "alts" in v

    def test_write_and_read_roundtrip(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.io_vcf import read_lrr, write_corrected

        lrr, samples, variants = read_lrr(test_bcf_path)
        info = {
            "k": 2,
            "backend": "rsvd",
            "n_hq_samples": BCF_N_SAMPLES,
            "n_markers_used": 1000,
            "singular_values": np.array([5.0, 2.0]),
        }
        out = tmp_path / "roundtrip.vcf"
        write_corrected(out, lrr, samples, variants, info)

        lrr2, samples2, variants2 = read_lrr(out)
        assert samples2 == samples
        assert lrr2.shape == lrr.shape
        valid = np.isfinite(lrr) & np.isfinite(lrr2)
        np.testing.assert_allclose(lrr2[valid], lrr[valid], atol=1e-4)


# ===================================================================
# 2. Genome build detection
# ===================================================================


class TestBuildDetection:
    """Verify genome build detection on real BCF."""

    def test_detects_build(self, test_bcf_path):
        from array_lrr_gwas.genome_build import detect_build

        build = detect_build(test_bcf_path)
        assert build == BCF_DETECTED_BUILD

    def test_exclusion_regions_for_detected_build(self, test_bcf_path):
        from array_lrr_gwas.genome_build import detect_build, get_exclusion_regions

        build = detect_build(test_bcf_path)
        regions = get_exclusion_regions(build)
        assert isinstance(regions, dict)
        assert "chr1" in regions
        assert all(
            isinstance(r, tuple) and len(r) == 2 for rs in regions.values() for r in rs
        )


# ===================================================================
# 3. Genotype extraction
# ===================================================================


class TestGenotypeExtraction:
    """Functional tests for genotype reading and filtering."""

    def test_dosage_shape_and_type(self, test_bcf_path):
        from array_lrr_gwas.genotypes import read_genotypes

        dosage, samples, variants = read_genotypes(
            test_bcf_path, min_maf=0.0, min_call_rate=0.0,
        )
        assert dosage.ndim == 2
        assert dosage.shape[1] == BCF_N_SAMPLES
        assert dosage.dtype == np.float64 or np.issubdtype(dosage.dtype, np.floating)

    def test_call_rate_filter(self, test_bcf_path):
        from array_lrr_gwas.genotypes import read_genotypes

        d_loose, _, _ = read_genotypes(test_bcf_path, min_maf=0.0, min_call_rate=0.0)
        d_strict, _, _ = read_genotypes(test_bcf_path, min_maf=0.0, min_call_rate=0.95)
        assert d_strict.shape[0] <= d_loose.shape[0]

    def test_maf_filter(self, test_bcf_path):
        from array_lrr_gwas.genotypes import read_genotypes

        d_all, _, _ = read_genotypes(test_bcf_path, min_maf=0.0, min_call_rate=0.0)
        d_maf, _, _ = read_genotypes(test_bcf_path, min_maf=0.10, min_call_rate=0.0)
        assert d_maf.shape[0] < d_all.shape[0]
        assert d_maf.shape[0] > 0


# ===================================================================
# 4. Marker subsetting on real data
# ===================================================================


class TestSubsettingOnBcf:
    """Marker QC filters applied to real BCF LRR data."""

    def test_call_rate_mask(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.subsetting import call_rate_mask

        lrr, _, _ = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        mask = call_rate_mask(lrr, min_call_rate=0.95)
        assert mask.shape == (BCF_N_VARIANTS,)
        # Most markers should pass with real data
        assert mask.sum() > BCF_N_VARIANTS * 0.9

    def test_variance_mask(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.subsetting import variance_mask

        lrr, _, _ = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        mask = variance_mask(lrr, min_var=0.001)
        assert mask.shape == (BCF_N_VARIANTS,)
        assert mask.sum() > 0

    def test_autosome_mask(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.subsetting import autosome_mask

        _, _, variants = read_lrr(test_bcf_path)
        chroms = np.array([v["chrom"] for v in variants])
        mask = autosome_mask(chroms)
        assert mask.shape == (BCF_N_VARIANTS,)
        # Autosomes should be the majority
        assert mask.sum() > BCF_N_VARIANTS * 0.9
        # Non-autosomal should be excluded
        # chrX markers should be excluded by autosome mask
        n_chrx = (chroms == "chrX").sum()
        if n_chrx > 0:
            assert not mask[chroms == "chrX"].any()

    def test_complexity_mask(self, test_bcf_path):
        from array_lrr_gwas.genome_build import detect_build, get_exclusion_regions
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.subsetting import complexity_mask

        _, _, variants = read_lrr(test_bcf_path)
        positions = np.array([v["pos"] for v in variants], dtype=np.intp)
        chroms = np.array([v["chrom"] for v in variants])
        build = detect_build(test_bcf_path)
        regions = get_exclusion_regions(build)

        mask = complexity_mask(positions, chroms, exclude_regions=regions)
        assert mask.shape == (BCF_N_VARIANTS,)
        # Some markers should be excluded (centromeres, etc.)
        assert mask.sum() < BCF_N_VARIANTS

    def test_subset_markers_combined(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.subsetting import subset_markers

        lrr, _, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        chroms = np.array([v["chrom"] for v in variants])
        positions = np.array([v["pos"] for v in variants], dtype=np.intp)

        mask = subset_markers(
            lrr,
            positions=positions,
            chromosomes=chroms,
            min_call_rate=0.90,
            min_var=0.001,
            autosomes_only=True,
        )
        assert mask.shape == (BCF_N_VARIANTS,)
        assert 0 < mask.sum() < BCF_N_VARIANTS


# ===================================================================
# 5. Correction pipeline on real data
# ===================================================================


class TestCorrectionOnBcf:
    """End-to-end batch correction on real BCF LRR values."""

    def test_classify_samples(self, test_bcf_path):
        from array_lrr_gwas.correction import classify_samples
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, _, _ = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        hq = classify_samples(lrr, max_lrr_sd=0.35)
        assert hq.shape == (BCF_N_SAMPLES,)
        # Real data has low LRR SD; all should be HQ with lenient threshold
        hq_lenient = classify_samples(lrr, max_lrr_sd=10.0, min_call_rate=0.0)
        assert hq_lenient.all()

    def test_correct_lrr_auto_k(self, test_bcf_path):
        from array_lrr_gwas.correction import correct_lrr
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, _, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        chroms = np.array([v["chrom"] for v in variants])
        positions = np.array([v["pos"] for v in variants], dtype=np.intp)

        corrected, info = correct_lrr(
            lrr,
            positions=positions,
            chromosomes=chroms,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        assert corrected.shape == lrr.shape
        assert info["k"] >= 1
        assert info["n_hq_samples"] == BCF_N_SAMPLES
        assert info["n_markers_used"] > 0

    def test_correct_lrr_fixed_k(self, test_bcf_path):
        from array_lrr_gwas.correction import correct_lrr
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, _, _ = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)

        corrected, info = correct_lrr(
            lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        assert info["k"] == 2
        assert corrected.shape == lrr.shape

    def test_correction_reduces_variance(self, test_bcf_path):
        from array_lrr_gwas.correction import correct_lrr
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, _, _ = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)

        corrected, _ = correct_lrr(
            lrr,
            k=3,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        orig_var = np.nanvar(lrr)
        corr_var = np.nanvar(corrected)
        assert corr_var < orig_var

    def test_correct_with_variant_qc_mask(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.correction import correct_lrr
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.variant_qc import read_collated_variant_qc, variant_qc_mask

        lrr, _, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)

        _write_variant_qc_tsv(tmp_path / "qc.tsv", variants, fail_first=True)
        qc_data = read_collated_variant_qc(tmp_path / "qc.tsv")
        vids = [v.get("id", "") for v in variants]
        mask = variant_qc_mask(vids, qc_data)

        corrected, info = correct_lrr(
            lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
            upstream_qc_mask=mask,
        )
        assert corrected.shape == lrr.shape


# ===================================================================
# 6. Decomposition on real data
# ===================================================================


class TestDecompositionOnBcf:
    """Truncated SVD on real LRR data."""

    def test_decompose_real_lrr(self, test_bcf_path):
        from array_lrr_gwas.decomposition import decompose
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.subsetting import subset_markers

        lrr, _, _ = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        mask = subset_markers(lrr, min_call_rate=0.90, min_var=0.001,
                              autosomes_only=False)
        sub = lrr[mask]

        k = 3
        U, s, Vt = decompose(sub, k=k)
        assert U.shape == (sub.shape[0], k)
        assert s.shape == (k,)
        assert Vt.shape == (k, BCF_N_SAMPLES)
        # Singular values should be positive and descending
        assert np.all(s > 0)
        assert np.all(np.diff(s) <= 0)

    def test_select_k_on_real_lrr(self, test_bcf_path):
        from array_lrr_gwas.decomposition import decompose
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.select_k import select_k_mp
        from array_lrr_gwas.subsetting import subset_markers

        lrr, _, _ = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        mask = subset_markers(lrr, min_call_rate=0.90, min_var=0.001,
                              autosomes_only=False)
        sub = lrr[mask]

        pilot_k = min(5, BCF_N_SAMPLES - 1)
        _, s, _ = decompose(sub, k=pilot_k)
        k = select_k_mp(s, n_markers=sub.shape[0], n_samples=sub.shape[1])
        assert k >= 1
        assert k <= pilot_k


# ===================================================================
# 7. GRM computation from real genotypes
# ===================================================================


class TestGrmOnBcf:
    """Compute GRM from real genotype data in the test BCF."""

    def test_compute_grm(self, test_bcf_path):
        from array_lrr_gwas.genotypes import read_genotypes
        from array_lrr_gwas.grm import compute_grm

        dosage, samples, _ = read_genotypes(
            test_bcf_path, min_maf=0.05, min_call_rate=0.90,
        )
        assert dosage.shape[0] > 0

        grm = compute_grm(dosage, min_maf=0.05)
        assert grm.shape == (BCF_N_SAMPLES, BCF_N_SAMPLES)
        # Symmetric
        np.testing.assert_allclose(grm, grm.T, atol=1e-10)
        # Positive semidefinite
        eigvals = np.linalg.eigvalsh(grm)
        assert np.all(eigvals >= -1e-6)

    def test_grm_diagonal(self, test_bcf_path):
        from array_lrr_gwas.genotypes import read_genotypes
        from array_lrr_gwas.grm import compute_grm

        dosage, _, _ = read_genotypes(
            test_bcf_path, min_maf=0.05, min_call_rate=0.90,
        )
        grm = compute_grm(dosage, min_maf=0.05)
        diag = np.diag(grm)
        # Diagonal should be close to 1 for unrelated samples
        np.testing.assert_allclose(diag, 1.0, atol=0.5)


# ===================================================================
# 8. LD pruning on real genotypes
# ===================================================================


class TestLdPruneOnBcf:
    """LD pruning on real genotype data."""

    def test_ld_prune_reduces_variants(self, test_bcf_path):
        from array_lrr_gwas.genotypes import read_genotypes
        from array_lrr_gwas.ld_prune import ld_prune

        dosage, _, variants = read_genotypes(
            test_bcf_path, min_maf=0.05, min_call_rate=0.90,
        )
        positions = np.array([v["pos"] for v in variants], dtype=np.intp)
        chroms = [v["chrom"] for v in variants]

        keep = ld_prune(
            dosage,
            positions=positions,
            chromosomes=chroms,
            r2_thresh=0.2,
            window_bp=1_000_000,
        )
        assert keep.shape == (dosage.shape[0],)
        assert keep.sum() < dosage.shape[0]
        assert keep.sum() > 0


# ===================================================================
# 9. Association scans with mock phenotypes
# ===================================================================


class TestAssociationOnBcf:
    """Run association scans against real BCF LRR with mock phenotypes."""

    def test_ols_scan(self, test_bcf_path):
        from array_lrr_gwas.association import run_association
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, samples, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        phenotype = _mock_phenotype(BCF_N_SAMPLES)

        result = run_association(lrr, phenotype, variants, method="ols")
        assert result.method == "ols"
        assert result.beta.shape == (BCF_N_VARIANTS,)
        assert result.se.shape == (BCF_N_VARIANTS,)
        finite_p = result.p_value[np.isfinite(result.p_value)]
        assert np.all((finite_p >= 0) & (finite_p <= 1))

    def test_ols_with_covariates(self, test_bcf_path):
        from array_lrr_gwas.association import run_association
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, _, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        rng = np.random.default_rng(42)
        phenotype = _mock_phenotype(BCF_N_SAMPLES)
        covariates = rng.normal(size=(BCF_N_SAMPLES, 2))

        result = run_association(
            lrr, phenotype, variants, method="ols", covariates=covariates,
        )
        assert result.beta.shape == (BCF_N_VARIANTS,)

    def test_logistic_scan(self, test_bcf_path):
        from array_lrr_gwas.association import run_association
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, _, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        phenotype = _mock_binary_phenotype(BCF_N_SAMPLES)

        result = run_association(lrr, phenotype, variants, method="logistic")
        assert result.method == "logistic"
        assert result.beta.shape == (BCF_N_VARIANTS,)

    def test_lmm_scan_with_real_grm(self, test_bcf_path):
        from array_lrr_gwas.association import run_association
        from array_lrr_gwas.genotypes import read_genotypes
        from array_lrr_gwas.grm import compute_grm
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, _, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)

        dosage, _, _ = read_genotypes(test_bcf_path, min_maf=0.05, min_call_rate=0.90)
        grm = compute_grm(dosage, min_maf=0.05)

        phenotype = _mock_phenotype(BCF_N_SAMPLES)
        result = run_association(
            lrr, phenotype, variants, method="lmm", grm=grm,
        )
        assert result.method == "lmm"
        assert result.beta.shape == (BCF_N_VARIANTS,)
        finite_p = result.p_value[np.isfinite(result.p_value)]
        assert np.all((finite_p >= 0) & (finite_p <= 1))

    def test_association_result_to_records(self, test_bcf_path):
        from array_lrr_gwas.association import run_association
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, _, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        phenotype = _mock_phenotype(BCF_N_SAMPLES)

        result = run_association(lrr, phenotype, variants, method="ols")
        records = result.to_records()
        assert isinstance(records, list)
        assert len(records) == BCF_N_VARIANTS
        r0 = records[0]
        assert "chrom" in r0
        assert "pos" in r0
        assert "beta" in r0
        assert "p_value" in r0
        assert "method" in r0


# ===================================================================
# 10. Segmentation on mock association output
# ===================================================================


class TestSegmentationOnBcf:
    """Segmentation of OLS association results derived from real BCF."""

    def test_hmm_segmentation(self, test_bcf_path):
        from array_lrr_gwas.association import run_association
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.segmentation import segment_associations

        lrr, _, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        phenotype = _mock_phenotype(BCF_N_SAMPLES)

        result = run_association(lrr, phenotype, variants, method="ols")
        records = result.to_records()

        seg = segment_associations(records, strategy="hmm")
        # With random phenotype, may get 0 or a few segments
        assert hasattr(seg, "chrom")
        assert hasattr(seg, "start")
        assert seg.strategy == "hmm"

    def test_threshold_segmentation(self, test_bcf_path):
        from array_lrr_gwas.association import run_association
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.segmentation import segment_associations

        lrr, _, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        phenotype = _mock_phenotype(BCF_N_SAMPLES)

        result = run_association(lrr, phenotype, variants, method="ols")
        records = result.to_records()

        # Use very lenient threshold to guarantee segments with random data
        seg = segment_associations(
            records, strategy="threshold", p_threshold=0.05, min_markers=1,
        )
        assert seg.strategy == "threshold"
        # Some markers should pass p < 0.05 by chance
        assert len(seg.chrom) > 0

    def test_segmentation_result_to_records(self, test_bcf_path):
        from array_lrr_gwas.association import run_association
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.segmentation import segment_associations

        lrr, _, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        phenotype = _mock_phenotype(BCF_N_SAMPLES)

        result = run_association(lrr, phenotype, variants, method="ols")
        records = result.to_records()

        seg = segment_associations(
            records, strategy="threshold", p_threshold=0.05, min_markers=1,
        )
        seg_records = seg.to_records()
        assert isinstance(seg_records, list)
        if len(seg_records) > 0:
            assert "chrom" in seg_records[0]
            assert "start" in seg_records[0]
            assert "end" in seg_records[0]

    def test_segmentation_write_bed(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.association import run_association
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.segmentation import segment_associations

        lrr, _, variants = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)
        phenotype = _mock_phenotype(BCF_N_SAMPLES)

        result = run_association(lrr, phenotype, variants, method="ols")
        records = result.to_records()

        seg = segment_associations(
            records, strategy="threshold", p_threshold=0.05, min_markers=1,
        )
        bed = tmp_path / "segments.bed"
        seg.write_bed(bed)
        assert bed.exists()
        content = bed.read_text()
        # Should have at least a header line
        assert len(content) > 0


# ===================================================================
# 11. Sample sheet alignment with BCF samples
# ===================================================================


class TestSampleSheetWithBcf:
    """Sample sheet parsing and alignment against BCF sample order."""

    def test_read_and_align(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.sample_sheet import align_samples, read_sample_sheet

        _, samples, _ = read_lrr(test_bcf_path)

        sheet = tmp_path / "sheet.tsv"
        _write_sample_sheet(sheet, samples, n_pcs=3)

        ids, covs, names = read_sample_sheet(sheet, n_pcs=3)
        assert ids == samples
        assert covs.shape == (BCF_N_SAMPLES, 3)

        aligned = align_samples(samples, ids, covs)
        assert aligned.shape == (BCF_N_SAMPLES, 3)
        np.testing.assert_array_equal(aligned, covs)

    def test_align_with_missing_samples(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.sample_sheet import align_samples, read_sample_sheet

        _, samples, _ = read_lrr(test_bcf_path)

        # Sheet only has first 4 samples
        sheet = tmp_path / "partial.tsv"
        _write_sample_sheet(sheet, samples[:4], n_pcs=2)

        ids, covs, names = read_sample_sheet(sheet, n_pcs=2)
        aligned = align_samples(samples, ids, covs)
        assert aligned.shape == (BCF_N_SAMPLES, 2)
        # First 4 should match, rest should be NaN
        assert not np.isnan(aligned[:4]).any()
        assert np.isnan(aligned[4:]).all()

    def test_classify_from_sheet(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.sample_sheet import classify_samples_from_sheet

        _, samples, _ = read_lrr(test_bcf_path)

        sheet = tmp_path / "qc_sheet.tsv"
        _write_sample_sheet(sheet, samples, include_qc=True)

        hq = classify_samples_from_sheet(
            sheet, max_lrr_sd=0.35, min_call_rate=0.95,
        )
        assert isinstance(hq, set)
        # All mock samples have call_rate ∈ [0.95, 1.0] and lrr_sd ∈ [0.05, 0.30]
        assert len(hq) == BCF_N_SAMPLES


# ===================================================================
# 12. Variant QC integration
# ===================================================================


class TestVariantQcWithBcf:
    """Variant QC mask applied to real BCF variant IDs."""

    def test_variant_qc_mask(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.variant_qc import read_collated_variant_qc, variant_qc_mask

        _, _, variants = read_lrr(test_bcf_path)
        _write_variant_qc_tsv(tmp_path / "qc.tsv", variants, fail_first=True)

        qc_data = read_collated_variant_qc(tmp_path / "qc.tsv")
        vids = [v.get("id", "") for v in variants]
        mask = variant_qc_mask(vids, qc_data)

        assert mask.shape == (BCF_N_VARIANTS,)
        # First variant should be masked out (fail_first=True)
        assert not mask[0]
        assert mask[1:].all()

    def test_variant_qc_all_pass(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.variant_qc import read_collated_variant_qc, variant_qc_mask

        _, _, variants = read_lrr(test_bcf_path)
        _write_variant_qc_tsv(tmp_path / "qc.tsv", variants, fail_first=False)

        qc_data = read_collated_variant_qc(tmp_path / "qc.tsv")
        vids = [v.get("id", "") for v in variants]
        mask = variant_qc_mask(vids, qc_data)
        assert mask.all()


# ===================================================================
# 13. QC config integration
# ===================================================================


class TestQcConfigWithBcf:
    """Apply QC config to correction of real BCF data."""

    def test_config_driven_correction(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.correction import correct_lrr
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.qc_config import apply_to_correct_args, defaults

        lrr, _, _ = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)

        cfg = defaults()
        kwargs = apply_to_correct_args(cfg, cli_overrides={
            "max_lrr_sd": 10.0,
            "min_sample_call_rate": 0.0,
            "min_marker_call_rate": 0.5,
            "min_var": 0.0,
            "k": 2,
        })
        corrected, info = correct_lrr(lrr, **kwargs)
        assert corrected.shape == lrr.shape
        assert info["k"] == 2

    def test_yaml_config_correction(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.correction import correct_lrr
        from array_lrr_gwas.io_vcf import read_lrr
        from array_lrr_gwas.qc_config import apply_to_correct_args, load_config

        lrr, _, _ = read_lrr(test_bcf_path)
        lrr = _sanitize_lrr(lrr)

        yaml_file = tmp_path / "qc.yaml"
        yaml_file.write_text(
            "sample_qc:\n"
            "  max_lrr_sd: 10.0\n"
            "  min_call_rate: 0.0\n"
            "marker_qc:\n"
            "  min_call_rate: 0.5\n"
            "  min_var: 0.0\n"
            "correction:\n"
            "  k: 2\n"
        )
        cfg = load_config(yaml_file)
        kwargs = apply_to_correct_args(cfg)
        corrected, info = correct_lrr(lrr, **kwargs)
        assert corrected.shape == lrr.shape
        assert info["k"] == 2


# ===================================================================
# 14. CLI integration tests
# ===================================================================


class TestCliIntegrationWithBcf:
    """End-to-end CLI subcommand tests against real BCF."""

    def test_correct_auto_build(self, test_bcf_path, tmp_path):
        """``correct`` sub-command with auto-detected build."""
        from array_lrr_gwas.cli import main

        out = tmp_path / "corrected.bcf"
        rc = main([
            "correct",
            str(test_bcf_path),
            "-o", str(out),
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
        ])
        assert rc == 0
        assert out.exists()

    def test_correct_explicit_build(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.cli import main

        out = tmp_path / "corrected.bcf"
        rc = main([
            "correct",
            str(test_bcf_path),
            "-o", str(out),
            "--build", "GRCh38",
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
        ])
        assert rc == 0
        assert out.exists()

    def test_correct_no_complexity(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.cli import main

        out = tmp_path / "corrected.vcf"
        rc = main([
            "correct",
            str(test_bcf_path),
            "-o", str(out),
            "--no-complexity-filter",
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
        ])
        assert rc == 0
        assert out.exists()

    def test_correct_with_variant_qc(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, _, variants = read_lrr(test_bcf_path)
        _write_variant_qc_tsv(tmp_path / "qc.tsv", variants)

        out = tmp_path / "corrected.bcf"
        rc = main([
            "correct",
            str(test_bcf_path),
            "-o", str(out),
            "--no-complexity-filter",
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
            "--variant-qc", str(tmp_path / "qc.tsv"),
        ])
        assert rc == 0
        assert out.exists()

    def test_associate_ols_cli(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(test_bcf_path)
        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(BCF_N_SAMPLES))

        out = tmp_path / "results.tsv"
        rc = main([
            "associate",
            str(test_bcf_path),
            "--phenotype", str(pheno),
            "--method", "ols",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert "chrom" in lines[0]
        assert len(lines) > 1

    def test_associate_logistic_cli(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(test_bcf_path)
        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_binary_phenotype(BCF_N_SAMPLES))

        out = tmp_path / "results.tsv"
        rc = main([
            "associate",
            str(test_bcf_path),
            "--phenotype", str(pheno),
            "--method", "logistic",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()

    def test_associate_lmm_cli(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(test_bcf_path)
        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(BCF_N_SAMPLES))

        out = tmp_path / "results.tsv"
        rc = main([
            "associate",
            str(test_bcf_path),
            "--phenotype", str(pheno),
            "--method", "lmm",
            "--ld-backend", "numpy",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert "chrom" in lines[0]
        assert len(lines) > 1

    def test_associate_ols_with_sample_sheet(self, test_bcf_path, tmp_path):
        """OLS with sample-sheet HQ filtering (covariates come from phenotype file)."""
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(test_bcf_path)
        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(BCF_N_SAMPLES))

        sheet = tmp_path / "sheet.tsv"
        _write_sample_sheet(sheet, samples, n_pcs=3, include_qc=True)

        out = tmp_path / "results.tsv"
        rc = main([
            "associate",
            str(test_bcf_path),
            "--phenotype", str(pheno),
            "--method", "ols",
            "--sample-sheet", str(sheet),
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()

    def test_segment_cli_after_association(self, test_bcf_path, tmp_path):
        """``segment`` sub-command on association results from real BCF."""
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(test_bcf_path)
        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(BCF_N_SAMPLES))

        assoc_out = tmp_path / "results.tsv"
        rc = main([
            "associate",
            str(test_bcf_path),
            "--phenotype", str(pheno),
            "--method", "ols",
            "-o", str(assoc_out),
        ])
        assert rc == 0

        seg_out = tmp_path / "segments.bed"
        rc = main([
            "segment",
            str(assoc_out),
            "-o", str(seg_out),
            "--strategy", "threshold",
            "--p-threshold", "0.05",
        ])
        assert rc == 0
        assert seg_out.exists()


# ===================================================================
# 15. Full pipeline: correct → associate → segment
# ===================================================================


class TestFullPipeline:
    """End-to-end pipeline test: correct → associate → segment."""

    def test_correct_then_associate_then_segment(self, test_bcf_path, tmp_path):
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(test_bcf_path)

        # Step 1: Correct
        corrected_bcf = tmp_path / "corrected.bcf"
        rc = main([
            "correct",
            str(test_bcf_path),
            "-o", str(corrected_bcf),
            "--no-complexity-filter",
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
            "--k", "2",
        ])
        assert rc == 0
        assert corrected_bcf.exists()

        # Step 2: Associate
        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(BCF_N_SAMPLES))

        assoc_out = tmp_path / "association.tsv"
        rc = main([
            "associate",
            str(corrected_bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "-o", str(assoc_out),
        ])
        assert rc == 0
        assert assoc_out.exists()
        lines = assoc_out.read_text().strip().split("\n")
        assert len(lines) > 1

        # Step 3: Segment
        seg_out = tmp_path / "segments.bed"
        rc = main([
            "segment",
            str(assoc_out),
            "-o", str(seg_out),
            "--strategy", "threshold",
            "--p-threshold", "0.05",
        ])
        assert rc == 0
        assert seg_out.exists()


# ===================================================================
# 16. Stage2 subsample BCF: associate as if pre-corrected
# ===================================================================


class TestStage2AssociatePipeline:
    """End-to-end associate tests using the stage2 100-sample subsample BCF.

    The stage2 BCF (``tests/data/stage2_reclustered.100.subsample.subset.bcf``)
    has 100 samples and ~12 k variants across all autosomes plus chrX/chrY.
    These tests exercise the ``associate`` sub-command directly on the raw BCF,
    treating it as if it were already RSVD/PC-corrected, which is the intended
    usage pattern when LRR has been corrected upstream.
    """

    def test_associate_ols_as_corrected(self, stage2_bcf_path, tmp_path):
        """OLS association on stage2 BCF, treating it as a pre-corrected input."""
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(stage2_bcf_path)
        assert len(samples) == STAGE2_N_SAMPLES

        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(STAGE2_N_SAMPLES))

        out = tmp_path / "results.tsv"
        rc = main([
            "associate",
            str(stage2_bcf_path),
            "--phenotype", str(pheno),
            "--method", "ols",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert "chrom" in lines[0]
        assert len(lines) > 1

    def test_associate_logistic_as_corrected(self, stage2_bcf_path, tmp_path):
        """Logistic regression on stage2 BCF, treating it as a pre-corrected input."""
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(stage2_bcf_path)

        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(
            pheno, samples, _mock_binary_phenotype(STAGE2_N_SAMPLES)
        )

        out = tmp_path / "results.tsv"
        rc = main([
            "associate",
            str(stage2_bcf_path),
            "--phenotype", str(pheno),
            "--method", "logistic",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert "chrom" in lines[0]
        assert len(lines) > 1

    def test_associate_lmm_as_corrected(self, stage2_bcf_path, tmp_path):
        """LMM (numpy LD backend) on stage2 BCF, treating it as a pre-corrected input."""
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(stage2_bcf_path)

        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(STAGE2_N_SAMPLES))

        out = tmp_path / "results.tsv"
        rc = main([
            "associate",
            str(stage2_bcf_path),
            "--phenotype", str(pheno),
            "--method", "lmm",
            "--ld-backend", "numpy",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert "chrom" in lines[0]
        assert len(lines) > 1

    @pytest.mark.skipif(
        not _plink2_available(),
        reason="plink2 not on PATH",
    )
    def test_associate_lmm_plink2_backend(self, stage2_bcf_path, tmp_path):
        """LMM with plink2 LD backend on stage2 BCF (plink2 GRM path)."""
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(stage2_bcf_path)

        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(STAGE2_N_SAMPLES))

        out = tmp_path / "results.tsv"
        rc = main([
            "associate",
            str(stage2_bcf_path),
            "--phenotype", str(pheno),
            "--method", "lmm",
            "--ld-backend", "plink2",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert "chrom" in lines[0]
        assert len(lines) > 1


    def test_associate_ols_with_sample_sheet_as_corrected(
        self, stage2_bcf_path, tmp_path
    ):
        """OLS with sample-sheet HQ filtering on stage2 BCF (pre-corrected input)."""
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(stage2_bcf_path)

        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(STAGE2_N_SAMPLES))

        sheet = tmp_path / "sheet.tsv"
        _write_sample_sheet(sheet, samples, n_pcs=3, include_qc=True)

        out = tmp_path / "results.tsv"
        rc = main([
            "associate",
            str(stage2_bcf_path),
            "--phenotype", str(pheno),
            "--method", "ols",
            "--sample-sheet", str(sheet),
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert "chrom" in lines[0]
        assert len(lines) > 1


class TestStage2AssociateSexChrBundled:
    """Regression guard for the sex-chromosome sex-resolution bug.

    Prior to the fix, running ``associate`` with the bundled
    ``tests/data/compiled_sample_sheet.tsv`` and
    ``tests/data/test_phenotype.tsv`` silently classified every sample
    as female — because the compiled sheet carries
    ``computed_gender`` / ``peddy_sex`` / ``f_sex`` rather than a
    numeric ``predicted_sex`` column, and the CLI only consulted the
    latter.  The fix wires up ``read_sample_sex_map``, which walks
    those columns with per-row fallback.  This integration test calls
    the CLI end-to-end against the bundled stage2 BCF, the compiled
    sample sheet, and the matching phenotype TSV, and asserts that
    both male and female strata are actually analysed.
    """

    def test_sex_chr_modes_use_bundled_compiled_sample_sheet(
        self, stage2_bcf_path, tmp_path, caplog,
    ) -> None:
        import logging
        from array_lrr_gwas.cli import main

        sheet = Path(__file__).parent / "data" / "compiled_sample_sheet.tsv"
        pheno = Path(__file__).parent / "data" / "test_phenotype.tsv"
        assert sheet.exists() and pheno.exists()

        out = tmp_path / "results.tsv"
        with caplog.at_level(logging.INFO, logger="array_lrr_gwas"):
            rc = main([
                "associate",
                str(stage2_bcf_path),
                "--phenotype", str(pheno),
                "--sample-sheet", str(sheet),
                "--method", "ols",
                "--sex-chr-mode",
                "x_with_sex_covariate",
                "x_male_only",
                "x_female_only",
                "y_male_only",
                "-o", str(out),
            ])
        assert rc == 0

        # The autosomal results file must be produced.
        assert out.exists()

        # Both male-only and female-only sidecar outputs must exist —
        # this is the specific regression we are guarding against.
        x_male_out = out.with_suffix("").with_name(
            out.stem + ".x_male_only" + out.suffix
        )
        x_female_out = out.with_suffix("").with_name(
            out.stem + ".x_female_only" + out.suffix
        )
        assert x_male_out.exists(), (
            "x_male_only sidecar not written — sex resolution likely "
            "still broken; log was:\n" + caplog.text
        )
        assert x_female_out.exists(), (
            "x_female_only sidecar not written — sex resolution likely "
            "still broken; log was:\n" + caplog.text
        )

        # The INFO log must report a non-zero count for BOTH sexes.
        # This asserts the underlying fix rather than just that the
        # sidecars happened to be created.
        breakdown_lines = [
            r.getMessage() for r in caplog.records
            if "sex breakdown" in r.getMessage()
        ]
        assert breakdown_lines, (
            "Expected a 'Sex-chr: analysed-cohort sex breakdown' log "
            "line; full log was:\n" + caplog.text
        )
        msg = breakdown_lines[0]
        # Expect "N male, M female" where both N and M > 0.
        import re
        m = re.search(r"(\d+) male, (\d+) female", msg)
        assert m, f"Unexpected breakdown format: {msg}"
        n_male, n_female = int(m.group(1)), int(m.group(2))
        assert n_male > 0, (
            f"Expected >0 males from bundled compiled sample sheet, got "
            f"{n_male}; log line: {msg}"
        )
        assert n_female > 0, (
            f"Expected >0 females from bundled compiled sample sheet, got "
            f"{n_female}; log line: {msg}"
        )

        # Neither the male-only nor the female-only mode should report
        # "Fewer than 3 samples" when both sexes are actually present.
        too_few_warnings = [
            r.getMessage() for r in caplog.records
            if "Fewer than 3 samples" in r.getMessage()
            and ("x_male_only" in r.getMessage()
                 or "x_female_only" in r.getMessage())
        ]
        assert not too_few_warnings, (
            "Unexpected 'Fewer than 3 samples' warnings for chrX "
            f"stratified modes: {too_few_warnings}"
        )




class TestStage2FullPipeline:
    """Full correct → associate (→ segment) pipeline on the stage2 subsample BCF.

    Exercises end-to-end plumbing: RSVD/PC correction on the raw stage2 BCF
    followed by OLS association, verifying that the corrected BCF produced by
    ``correct`` is a valid input to ``associate``.
    """

    def test_correct_then_associate(self, stage2_bcf_path, tmp_path):
        """correct → associate (OLS) pipeline on stage2 BCF."""
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(stage2_bcf_path)

        # Step 1: Correct LRR (RSVD/PC correction)
        corrected_bcf = tmp_path / "corrected.bcf"
        rc = main([
            "correct",
            str(stage2_bcf_path),
            "-o", str(corrected_bcf),
            "--no-complexity-filter",
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
            "--k", "3",
        ])
        assert rc == 0
        assert corrected_bcf.exists()

        # Step 2: Associate on the corrected BCF
        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(STAGE2_N_SAMPLES))

        assoc_out = tmp_path / "association.tsv"
        rc = main([
            "associate",
            str(corrected_bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "-o", str(assoc_out),
        ])
        assert rc == 0
        assert assoc_out.exists()
        lines = assoc_out.read_text().strip().split("\n")
        assert "chrom" in lines[0]
        assert len(lines) > 1

    def test_full_pipeline_correct_associate_segment(
        self, stage2_bcf_path, tmp_path
    ):
        """Full correct → associate → segment pipeline on stage2 BCF."""
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(stage2_bcf_path)

        # Step 1: Correct
        corrected_bcf = tmp_path / "corrected.bcf"
        rc = main([
            "correct",
            str(stage2_bcf_path),
            "-o", str(corrected_bcf),
            "--no-complexity-filter",
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
            "--k", "3",
        ])
        assert rc == 0
        assert corrected_bcf.exists()

        # Step 2: Associate
        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(STAGE2_N_SAMPLES))

        assoc_out = tmp_path / "association.tsv"
        rc = main([
            "associate",
            str(corrected_bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "-o", str(assoc_out),
        ])
        assert rc == 0
        assert assoc_out.exists()

        # Step 3: Segment
        seg_out = tmp_path / "segments.bed"
        rc = main([
            "segment",
            str(assoc_out),
            "-o", str(seg_out),
            "--strategy", "threshold",
            "--p-threshold", "0.05",
        ])
        assert rc == 0
        assert seg_out.exists()

    @pytest.mark.skipif(
        not _plink2_available(),
        reason="plink2 not on PATH",
    )
    def test_correct_then_associate_lmm_plink2(self, stage2_bcf_path, tmp_path):
        """correct → associate (LMM, plink2 GRM) pipeline on stage2 BCF."""
        from array_lrr_gwas.cli import main
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(stage2_bcf_path)

        # Step 1: Correct LRR (RSVD/PC correction)
        corrected_bcf = tmp_path / "corrected.bcf"
        rc = main([
            "correct",
            str(stage2_bcf_path),
            "-o", str(corrected_bcf),
            "--no-complexity-filter",
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
            "--k", "3",
        ])
        assert rc == 0
        assert corrected_bcf.exists()

        # Step 2: Associate using the plink2 GRM path.
        # The corrected BCF only has LRR — supply the original BCF via
        # --genotype-bcf so that plink2 can build the GRM from GT data.
        pheno = tmp_path / "pheno.tsv"
        _write_phenotype_tsv(pheno, samples, _mock_phenotype(STAGE2_N_SAMPLES))

        assoc_out = tmp_path / "association.tsv"
        rc = main([
            "associate",
            str(corrected_bcf),
            "--phenotype", str(pheno),
            "--method", "lmm",
            "--ld-backend", "plink2",
            "--genotype-bcf", str(stage2_bcf_path),
            "-o", str(assoc_out),
        ])
        assert rc == 0
        assert assoc_out.exists()
        lines = assoc_out.read_text().strip().split("\n")
        assert "chrom" in lines[0]
        assert len(lines) > 1
