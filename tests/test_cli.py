"""Tests for the CLI entry point."""

from pathlib import Path

import numpy as np
import pytest

from array_lrr_gwas.cli import _build_parser, _variant_id, main
from conftest import mock_associate_io


class TestCli:
    def test_variant_id_handles_none_alts(self):
        vid = _variant_id({"chrom": "chr1", "pos": 123, "id": ".", "ref": "A", "alts": None})
        assert vid == "chr1:123:A:"

    def test_associate_ld_backend_default_is_plink2(self):
        args = _build_parser().parse_args([
            "associate",
            "in.bcf",
            "--phenotype", "pheno.tsv",
            "-o", "out.tsv",
        ])
        assert args.ld_backend == "plink2"

    def test_correct_variant_qc_arg_parsed(self):
        args = _build_parser().parse_args([
            "correct",
            "in.bcf",
            "-o", "out.bcf",
            "--variant-qc", "/data/collated_variant_qc.tsv",
        ])
        assert args.variant_qc == Path("/data/collated_variant_qc.tsv")

    def test_correct_variant_qc_default_none(self):
        args = _build_parser().parse_args([
            "correct",
            "in.bcf",
            "-o", "out.bcf",
        ])
        assert args.variant_qc is None

    def test_correct_n_components_arg_parsed(self):
        args = _build_parser().parse_args([
            "correct",
            "in.bcf",
            "-o", "out.bcf",
            "--n-components", "12",
        ])
        assert args.n_components == 12

    def test_correct_svd_output_args_parsed(self):
        args = _build_parser().parse_args([
            "correct",
            "in.bcf",
            "-o", "out.bcf",
            "--svd-output-prefix", "my_prefix",
            "--write-loadings",
        ])
        assert args.svd_output_prefix == Path("my_prefix")
        assert args.write_loadings is True

    def test_associate_variant_qc_arg_parsed(self):
        args = _build_parser().parse_args([
            "associate",
            "in.bcf",
            "--phenotype", "pheno.tsv",
            "-o", "out.tsv",
            "--variant-qc", "/data/collated_variant_qc.tsv",
        ])
        assert args.variant_qc == Path("/data/collated_variant_qc.tsv")

    def test_associate_variant_qc_default_none(self):
        args = _build_parser().parse_args([
            "associate",
            "in.bcf",
            "--phenotype", "pheno.tsv",
            "-o", "out.tsv",
        ])
        assert args.variant_qc is None

    def test_associate_config_arg_parsed(self):
        args = _build_parser().parse_args([
            "associate",
            "in.bcf",
            "--phenotype", "pheno.tsv",
            "-o", "out.tsv",
            "--config", "/data/qc.yaml",
        ])
        assert args.config == Path("/data/qc.yaml")

    def test_associate_hq_samples_arg_parsed(self):
        args = _build_parser().parse_args([
            "associate",
            "in.bcf",
            "--phenotype", "pheno.tsv",
            "-o", "out.tsv",
            "--hq-samples", "/data/hq_samples.txt",
        ])
        assert args.hq_samples == Path("/data/hq_samples.txt")

    def test_associate_max_lrr_sd_arg_parsed(self):
        args = _build_parser().parse_args([
            "associate",
            "in.bcf",
            "--phenotype", "pheno.tsv",
            "-o", "out.tsv",
            "--max-lrr-sd", "0.30",
        ])
        assert args.max_lrr_sd == 0.30

    def test_associate_min_sample_call_rate_arg_parsed(self):
        args = _build_parser().parse_args([
            "associate",
            "in.bcf",
            "--phenotype", "pheno.tsv",
            "-o", "out.tsv",
            "--min-sample-call-rate", "0.98",
        ])
        assert args.min_sample_call_rate == 0.98

    def test_associate_max_lrr_sd_default_none(self):
        args = _build_parser().parse_args([
            "associate",
            "in.bcf",
            "--phenotype", "pheno.tsv",
            "-o", "out.tsv",
        ])
        assert args.max_lrr_sd is None
        assert args.min_sample_call_rate is None

    def test_no_args_returns_1(self):
        assert main([]) == 1

    def test_help_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["correct", "--help"])
        assert exc.value.code == 0

    def test_correct_missing_input(self, tmp_path):
        out = tmp_path / "out.bcf"
        rc = main(["correct", str(tmp_path / "nonexistent.bcf"), "-o", str(out)])
        assert rc == 1

    def test_correct_build_auto_detected(self, test_bcf_path, tmp_path):
        """Build is auto-detected from contig lengths in the new test BCF."""
        out = tmp_path / "out.bcf"
        rc = main([
            "correct",
            str(test_bcf_path),
            "-o", str(out),
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
        ])
        # Build detection succeeds (T2T-CHM13), so correction should succeed
        assert rc == 0
        assert out.exists()

    def test_correct_with_explicit_build(self, test_bcf_path, tmp_path):
        """Correction succeeds when build is explicitly given."""
        out = tmp_path / "out.bcf"
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

    def test_correct_with_t2t_build(self, test_bcf_path, tmp_path):
        """T2T-CHM13 is accepted as a build argument."""
        out = tmp_path / "out.vcf"
        rc = main([
            "correct",
            str(test_bcf_path),
            "-o", str(out),
            "--build", "T2T-CHM13",
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
        ])
        assert rc == 0
        assert out.exists()

    def test_correct_no_complexity_filter(self, test_bcf_path, tmp_path):
        """--no-complexity-filter skips build requirement."""
        out = tmp_path / "out.bcf"
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

    def test_correct_with_build_alias(self, test_bcf_path, tmp_path):
        """Common aliases (hg19, hs1, etc.) are accepted."""
        out = tmp_path / "out.bcf"
        rc = main([
            "correct",
            str(test_bcf_path),
            "-o", str(out),
            "--build", "hs1",
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
        ])
        assert rc == 0
        assert out.exists()

    def test_correct_with_variant_qc(self, test_bcf_path, tmp_path):
        """--variant-qc applies upstream QC mask during correction."""
        from array_lrr_gwas.io_vcf import read_lrr

        _, _, variants = read_lrr(test_bcf_path)
        # Build a QC TSV with the variant IDs from the test BCF.
        # Mark the first variant as failing call rate to verify masking.
        qc_tsv = tmp_path / "collated_variant_qc.tsv"
        lines = [
            "variant_id\tall_ancestries_call_rate_pass\t"
            "all_ancestries_hwe_pass\tall_ancestries_maf_pass"
        ]
        for i, v in enumerate(variants):
            vid = v.get("id") or f"{v['chrom']}:{v['pos']}:{v.get('ref', '')}:{':'.join(v.get('alts', ()))}"
            cr = "False" if i == 0 else "True"
            lines.append(f"{vid}\t{cr}\tTrue\tTrue")
        qc_tsv.write_text("\n".join(lines) + "\n")

        out = tmp_path / "out.bcf"
        rc = main([
            "correct",
            str(test_bcf_path),
            "-o", str(out),
            "--no-complexity-filter",
            "--max-lrr-sd", "10.0",
            "--min-sample-call-rate", "0.0",
            "--min-marker-call-rate", "0.5",
            "--min-var", "0.0",
            "--variant-qc", str(qc_tsv),
        ])
        assert rc == 0
        assert out.exists()

    def test_correct_writes_svd_text_outputs(self, tmp_path, monkeypatch):
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")
        out = tmp_path / "out.bcf"

        lrr = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
        samples = ["S1", "S2"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "v1", "ref": "A", "alts": ("C",)},
            {"chrom": "chr1", "pos": 200, "id": ".", "ref": "G", "alts": ("T",)},
        ]

        info = {
            "k": 2,
            "n_components_computed": 2,
            "singular_values": np.array([3.0, 1.5], dtype=float),
            "sample_scores": np.array([[0.10, 0.20], [0.30, 0.40]], dtype=float),
            "marker_loadings": np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float),
            "marker_mask": np.array([True, True], dtype=bool),
            "hq_sample_mask": np.array([True, True], dtype=bool),
            "n_hq_samples": 2,
            "n_markers_used": 2,
            "backend": "rsvd",
        }

        monkeypatch.setattr("array_lrr_gwas.io_vcf.read_lrr", lambda _p: (lrr, samples, variants))
        monkeypatch.setattr("array_lrr_gwas.io_vcf.write_corrected", lambda *_a, **_k: None)
        monkeypatch.setattr("array_lrr_gwas.correction.correct_lrr", lambda *_a, **_k: (lrr, info))

        rc = main([
            "correct",
            str(fake_bcf),
            "-o", str(out),
            "--no-complexity-filter",
        ])
        assert rc == 0

        pcs_path = tmp_path / "out.bcf.svd.sample_pcs.tsv"
        sv_path = tmp_path / "out.bcf.svd.singular_values.tsv"
        assert pcs_path.exists()
        assert sv_path.exists()

        pcs_lines = pcs_path.read_text().strip().splitlines()
        assert pcs_lines[0] == "SAMPLE\tPC1\tPC2"
        assert pcs_lines[1] == "S1\t0.3\t0.45"
        assert pcs_lines[2] == "S2\t0.6\t0.6"

        sv_lines = sv_path.read_text().strip().splitlines()
        assert sv_lines == [
            "PC\tsingular_value\tused_for_correction",
            "PC1\t3\tyes",
            "PC2\t1.5\tyes",
        ]

    def test_correct_writes_svd_outputs_pilot_gt_k(self, tmp_path, monkeypatch):
        """When n_components_computed > k, all pilot PCs are written and
        only the first k are marked used_for_correction=yes."""
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")
        out = tmp_path / "out.bcf"

        lrr = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
        samples = ["S1", "S2"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "v1", "ref": "A", "alts": ("C",)},
            {"chrom": "chr1", "pos": 200, "id": ".", "ref": "G", "alts": ("T",)},
        ]
        # k=1 selected but 3 were computed
        info = {
            "k": 1,
            "n_components_computed": 3,
            "singular_values": np.array([5.0, 2.0, 0.5], dtype=float),
            "sample_scores": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=float),
            "marker_loadings": np.array([[0.9, 0.1, 0.05], [0.2, 0.8, 0.02]], dtype=float),
            "marker_mask": np.array([True, True], dtype=bool),
            "hq_sample_mask": np.array([True, True], dtype=bool),
            "n_hq_samples": 2,
            "n_markers_used": 2,
            "backend": "rsvd",
        }

        monkeypatch.setattr("array_lrr_gwas.io_vcf.read_lrr", lambda _p: (lrr, samples, variants))
        monkeypatch.setattr("array_lrr_gwas.io_vcf.write_corrected", lambda *_a, **_k: None)
        monkeypatch.setattr("array_lrr_gwas.correction.correct_lrr", lambda *_a, **_k: (lrr, info))

        rc = main([
            "correct",
            str(fake_bcf),
            "-o", str(out),
            "--no-complexity-filter",
        ])
        assert rc == 0

        sv_path = tmp_path / "out.bcf.svd.singular_values.tsv"
        sv_lines = sv_path.read_text().strip().splitlines()
        assert sv_lines == [
            "PC\tsingular_value\tused_for_correction",
            "PC1\t5\tyes",
            "PC2\t2\tno",
            "PC3\t0.5\tno",
        ]

        pcs_path = tmp_path / "out.bcf.svd.sample_pcs.tsv"
        pcs_lines = pcs_path.read_text().strip().splitlines()
        # Header should contain all 3 PCs
        assert pcs_lines[0] == "SAMPLE\tPC1\tPC2\tPC3"

    def test_correct_writes_loadings_when_requested(self, tmp_path, monkeypatch):
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")
        out = tmp_path / "out.bcf"
        custom_prefix = tmp_path / "svd_summary"

        lrr = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
        samples = ["S1", "S2"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "v1", "ref": "A", "alts": ("C",)},
            {"chrom": "chr1", "pos": 200, "id": ".", "ref": "G", "alts": ("T",)},
        ]
        info = {
            "k": 2,
            "n_components_computed": 2,
            "singular_values": np.array([3.0, 1.5], dtype=float),
            "sample_scores": np.array([[0.10, 0.20], [0.30, 0.40]], dtype=float),
            "marker_loadings": np.array([[0.9, 0.1]], dtype=float),
            "marker_mask": np.array([True, False], dtype=bool),
            "hq_sample_mask": np.array([True, True], dtype=bool),
            "n_hq_samples": 2,
            "n_markers_used": 1,
            "backend": "rsvd",
        }

        monkeypatch.setattr("array_lrr_gwas.io_vcf.read_lrr", lambda _p: (lrr, samples, variants))
        monkeypatch.setattr("array_lrr_gwas.io_vcf.write_corrected", lambda *_a, **_k: None)
        monkeypatch.setattr("array_lrr_gwas.correction.correct_lrr", lambda *_a, **_k: (lrr, info))

        rc = main([
            "correct",
            str(fake_bcf),
            "-o", str(out),
            "--no-complexity-filter",
            "--svd-output-prefix", str(custom_prefix),
            "--write-loadings",
        ])
        assert rc == 0

        loadings_path = tmp_path / "svd_summary.loadings.tsv"
        assert loadings_path.exists()
        loadings_lines = loadings_path.read_text().strip().splitlines()
        assert loadings_lines == [
            "chrom\tpos\tvariant_id\tPC1\tPC2",
            "chr1\t100\tv1\t0.9\t0.1",
        ]

    def test_associate_help_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["associate", "--help"])
        assert exc.value.code == 0

    def test_associate_missing_input(self, tmp_path):
        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.5\n")
        out = tmp_path / "results.tsv"
        rc = main([
            "associate",
            str(tmp_path / "nonexistent.bcf"),
            "--phenotype", str(pheno),
            "-o", str(out),
        ])
        assert rc == 1

    def test_associate_ols(self, test_bcf_path, tmp_path):
        """OLS association via CLI with test BCF."""
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(test_bcf_path)

        # Write phenotype TSV
        import numpy as np

        rng = np.random.default_rng(42)
        pheno = tmp_path / "pheno.tsv"
        lines = ["sample_id\tphenotype"]
        for s in samples:
            lines.append(f"{s}\t{rng.normal():.6f}")
        pheno.write_text("\n".join(lines) + "\n")

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
        # Verify output has correct header and data rows
        content = out.read_text().strip().split("\n")
        assert "chrom" in content[0]
        assert len(content) > 1

    def test_associate_lmm_succeeds_with_gt(self, test_bcf_path, tmp_path):
        """LMM with a BCF that has GT data should succeed."""
        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(test_bcf_path)

        import numpy as np

        rng = np.random.default_rng(42)
        pheno = tmp_path / "pheno.tsv"
        lines = ["sample_id\tphenotype"]
        for s in samples:
            lines.append(f"{s}\t{rng.normal():.6f}")
        pheno.write_text("\n".join(lines) + "\n")

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
        content = out.read_text().strip().split("\n")
        assert "chrom" in content[0]
        assert len(content) > 1

    def test_associate_cli_warns_when_logistic_ignores_grm(
        self, test_bcf_path, tmp_path, caplog
    ):
        """CLI should warn that logistic does not use the GRM random effect."""
        import logging
        import numpy as np

        from array_lrr_gwas.io_vcf import read_lrr

        _, samples, _ = read_lrr(test_bcf_path)
        rng = np.random.default_rng(42)
        pheno = tmp_path / "pheno.tsv"
        lines = ["sample_id\tphenotype"]
        for s in samples:
            y = 1 if rng.random() > 0.5 else 0
            lines.append(f"{s}\t{y}")
        pheno.write_text("\n".join(lines) + "\n")

        out = tmp_path / "results.tsv"
        with caplog.at_level(logging.WARNING, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(test_bcf_path),
                "--phenotype", str(pheno),
                "--method", "logistic",
                "-o", str(out),
            ])
        assert rc == 0
        assert out.exists()
        assert "does not use the GRM random effect" in caplog.text

    def test_associate_lmm_plink2_fallbacks_to_numpy(
        self, tmp_path, monkeypatch, caplog
    ):
        """If plink2 is unavailable, CLI should fall back to NumPy pruning."""
        import logging

        from array_lrr_gwas import association

        # Minimal phenotype file
        pheno = tmp_path / "pheno.tsv"
        pheno.write_text(
            "sample_id\tphenotype\n"
            "S1\t0.1\nS2\t0.2\nS3\t0.3\n"
        )
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        # LRR input for association
        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        # GT input for GRM
        dosage = np.array(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
            dtype=float,
        )
        gt_variants = [
            {"chrom": "chr1", "pos": 100, "id": "v1", "ref": "A", "alts": ("C",)},
            {"chrom": "chr1", "pos": 110, "id": "v2", "ref": "A", "alts": ("G",)},
            {"chrom": "chr1", "pos": 120, "id": "v3", "ref": "A", "alts": ("T",)},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.genotypes.read_genotypes",
            lambda *_a, **_k: (dosage, samples, gt_variants),
        )

        def _raise_plink2_missing(*_a, **_k):
            raise FileNotFoundError("plink2")

        monkeypatch.setattr(
            "array_lrr_gwas.ld_prune.ld_prune_plink2",
            _raise_plink2_missing,
        )
        monkeypatch.setattr(
            "array_lrr_gwas.ld_prune.ld_prune",
            lambda *_a, **_k: np.array([True, False, True], dtype=bool),
        )

        def _fake_grm(d, *, min_maf=0.01):
            assert d.shape[0] == 2  # pruned from 3 to 2
            return np.eye(d.shape[1], dtype=float)

        monkeypatch.setattr("array_lrr_gwas.grm.compute_grm", _fake_grm)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {
                        "chrom": "chr1",
                        "pos": 100,
                        "variant_id": "a1",
                        "beta": 0.0,
                        "se": 1.0,
                        "stat": 0.0,
                        "p_value": 1.0,
                        "n_samples": 3,
                        "method": "lmm",
                    },
                    {
                        "chrom": "chr1",
                        "pos": 200,
                        "variant_id": "a2",
                        "beta": 0.0,
                        "se": 1.0,
                        "stat": 0.0,
                        "p_value": 1.0,
                        "n_samples": 3,
                        "method": "lmm",
                    },
                ]

        monkeypatch.setattr(
            association,
            "run_association_streaming",
            lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}),
        )

        with caplog.at_level(logging.WARNING, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--method", "lmm",
                "-o", str(out),
            ])

        assert rc == 0
        assert out.exists()
        assert "falling back to NumPy LD pruning" in caplog.text

    def test_associate_lmm_with_variant_qc(self, tmp_path, monkeypatch, caplog):
        """--variant-qc applies upstream QC mask to GRM markers before LD."""
        import logging

        from array_lrr_gwas import association

        # Write a QC TSV: v1 passes, v2 fails MAF, v3 passes
        qc_tsv = tmp_path / "collated_variant_qc.tsv"
        qc_tsv.write_text(
            "variant_id\tall_ancestries_call_rate_pass\t"
            "all_ancestries_hwe_pass\tall_ancestries_maf_pass\n"
            "v1\tTrue\tTrue\tTrue\n"
            "v2\tTrue\tTrue\tFalse\n"
            "v3\tTrue\tTrue\tTrue\n"
        )

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text(
            "sample_id\tphenotype\n"
            "S1\t0.1\nS2\t0.2\nS3\t0.3\n"
        )
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        # 3 GT variants; v2 will be removed by QC mask
        dosage = np.array(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
            dtype=float,
        )
        gt_variants = [
            {"chrom": "chr1", "pos": 100, "id": "v1", "ref": "A", "alts": ("C",)},
            {"chrom": "chr1", "pos": 110, "id": "v2", "ref": "A", "alts": ("G",)},
            {"chrom": "chr1", "pos": 120, "id": "v3", "ref": "A", "alts": ("T",)},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.genotypes.read_genotypes",
            lambda *_a, **_k: (dosage.copy(), samples, list(gt_variants)),
        )

        # NumPy LD pruning — receives only 2 variants (after QC mask removes v2)
        def _fake_ld_prune(*_a, **_k):
            # Should receive a dosage array with 2 rows (v1 + v3)
            d = _a[0]
            return np.ones(d.shape[0], dtype=bool)  # keep all surviving

        monkeypatch.setattr(
            "array_lrr_gwas.ld_prune.ld_prune",
            _fake_ld_prune,
        )

        def _fake_grm(d, *, min_maf=0.01):
            # After QC mask, should have 2 variants
            assert d.shape[0] == 2, f"Expected 2 variants after QC, got {d.shape[0]}"
            return np.eye(d.shape[1], dtype=float)

        monkeypatch.setattr("array_lrr_gwas.grm.compute_grm", _fake_grm)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "lmm"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "lmm"},
                ]

        monkeypatch.setattr(
            association,
            "run_association_streaming",
            lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}),
        )

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--method", "lmm",
                "--ld-backend", "numpy",
                "--variant-qc", str(qc_tsv),
                "-o", str(out),
            ])

        assert rc == 0
        assert out.exists()
        assert "Upstream variant QC (GRM)" in caplog.text

    def test_associate_lmm_variant_qc_from_config(self, tmp_path, monkeypatch):
        """When --variant-qc is unset, associate reads upstream_qc.variant_qc_path from --config."""
        from array_lrr_gwas import association

        qc_tsv = tmp_path / "collated_variant_qc.tsv"
        qc_tsv.write_text(
            "variant_id\tall_ancestries_call_rate_pass\t"
            "all_ancestries_hwe_pass\tall_ancestries_maf_pass\n"
            "v1\tTrue\tTrue\tTrue\n"
            "v2\tTrue\tTrue\tFalse\n"
            "v3\tTrue\tTrue\tTrue\n"
        )
        cfg = tmp_path / "qc.yaml"
        cfg.write_text(f"upstream_qc:\n  variant_qc_path: {qc_tsv}\n")

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        assoc_variants = [{"chrom": "chr1", "pos": 100, "id": "a1"}, {"chrom": "chr1", "pos": 200, "id": "a2"}]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        dosage = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=float)
        gt_variants = [
            {"chrom": "chr1", "pos": 100, "id": "v1", "ref": "A", "alts": ("C",)},
            {"chrom": "chr1", "pos": 110, "id": "v2", "ref": "A", "alts": ("G",)},
            {"chrom": "chr1", "pos": 120, "id": "v3", "ref": "A", "alts": ("T",)},
        ]
        monkeypatch.setattr("array_lrr_gwas.genotypes.read_genotypes", lambda *_a, **_k: (dosage.copy(), samples, list(gt_variants)))
        monkeypatch.setattr("array_lrr_gwas.ld_prune.ld_prune", lambda d, **_k: np.ones(d.shape[0], dtype=bool))
        monkeypatch.setattr("array_lrr_gwas.grm.compute_grm", lambda d, **_k: np.eye(d.shape[1], dtype=float))

        class _FakeResult:
            variant_id = ['a1']
            chrom = ["chr1"]
            p_value = np.array([1.0])
            stat = np.array([0.0])
            beta = np.array([0.0])
            se = np.array([1.0])

            @staticmethod
            def to_records():
                return [{
                    "chrom": "chr1", "pos": 100, "variant_id": "a1",
                    "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                    "n_samples": 3, "method": "lmm",
                }]

        monkeypatch.setattr(association, "run_association_streaming", lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}))

        rc = main([
            "associate",
            str(fake_bcf),
            "--phenotype", str(pheno),
            "--method", "lmm",
            "--ld-backend", "numpy",
            "--config", str(cfg),
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()

    def test_associate_qc_provenance_columns_in_output(self, tmp_path, monkeypatch):
        """When --variant-qc is set, output TSV includes QC provenance columns."""
        import csv

        from array_lrr_gwas import association

        # QC TSV with flags for the LRR variant IDs
        qc_tsv = tmp_path / "collated_variant_qc.tsv"
        qc_tsv.write_text(
            "variant_id\tall_ancestries_call_rate_pass\t"
            "all_ancestries_hwe_pass\tall_ancestries_maf_pass\n"
            "a1\tTrue\tTrue\tFalse\n"
            "a2\tFalse\tTrue\tTrue\n"
        )

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                ]

        monkeypatch.setattr(
            association, "run_association_streaming",
            lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}),
        )

        rc = main([
            "associate",
            str(fake_bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "--variant-qc", str(qc_tsv),
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()

        with open(out, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)

        # Check QC columns are present
        assert "all_ancestries_call_rate_pass" in reader.fieldnames
        assert "all_ancestries_hwe_pass" in reader.fieldnames
        assert "all_ancestries_maf_pass" in reader.fieldnames
        assert "all_ancestries_qc_pass" in reader.fieldnames

        # Check values for a1
        assert rows[0]["all_ancestries_call_rate_pass"] == "True"
        assert rows[0]["all_ancestries_hwe_pass"] == "True"
        assert rows[0]["all_ancestries_maf_pass"] == "False"
        assert rows[0]["all_ancestries_qc_pass"] == ""

        # Check values for a2
        assert rows[1]["all_ancestries_call_rate_pass"] == "False"
        assert rows[1]["all_ancestries_hwe_pass"] == "True"
        assert rows[1]["all_ancestries_maf_pass"] == "True"
        assert rows[1]["all_ancestries_qc_pass"] == ""

    def test_associate_propagates_composite_variant_qc_flag(self, tmp_path, monkeypatch):
        """When present upstream, all_ancestries_qc_pass is propagated to output."""
        import csv

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")
        qc_tsv = tmp_path / "qc.tsv"
        qc_tsv.write_text(
            "variant_id\tall_ancestries_call_rate_pass\t"
            "all_ancestries_hwe_pass\tall_ancestries_maf_pass\t"
            "all_ancestries_qc_pass\n"
            "a1\tTrue\tTrue\tFalse\tFalse\n"
            "a2\tFalse\tTrue\tTrue\tFalse\n"
        )

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                ]

        monkeypatch.setattr(
            association, "run_association_streaming",
            lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}),
        )

        rc = main([
            "associate",
            str(fake_bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "--variant-qc", str(qc_tsv),
            "-o", str(out),
        ])
        assert rc == 0

        with open(out, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)

        assert rows[0]["all_ancestries_qc_pass"] == "False"
        assert rows[1]["all_ancestries_qc_pass"] == "False"

    def test_associate_no_qc_provenance_without_variant_qc(self, tmp_path, monkeypatch):
        """Without --variant-qc, output TSV should NOT include QC columns."""
        import csv

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                ]

        monkeypatch.setattr(
            association, "run_association_streaming",
            lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}),
        )

        rc = main([
            "associate",
            str(fake_bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()

        with open(out, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            list(reader)

        assert "all_ancestries_call_rate_pass" not in reader.fieldnames

    def test_associate_logs_warning_when_no_variant_qc(
        self, tmp_path, monkeypatch, caplog
    ):
        """When no --variant-qc is set, a warning is logged about missing upstream variant QC."""
        import logging

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        dosage = np.array(
            [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=float,
        )
        gt_variants = [
            {"chrom": "chr1", "pos": 100, "id": "v1", "ref": "A", "alts": ("C",)},
            {"chrom": "chr1", "pos": 120, "id": "v3", "ref": "A", "alts": ("T",)},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.genotypes.read_genotypes",
            lambda *_a, **_k: (dosage.copy(), samples, list(gt_variants)),
        )
        monkeypatch.setattr(
            "array_lrr_gwas.ld_prune.ld_prune",
            lambda d, **_k: np.ones(d.shape[0], dtype=bool),
        )
        monkeypatch.setattr(
            "array_lrr_gwas.grm.compute_grm",
            lambda d, **_k: np.eye(d.shape[1], dtype=float),
        )

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "lmm"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "lmm"},
                ]

        monkeypatch.setattr(
            association, "run_association_streaming",
            lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}),
        )

        with caplog.at_level(logging.WARNING, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--method", "lmm",
                "--ld-backend", "numpy",
                "-o", str(out),
            ])

        assert rc == 0
        assert "No upstream variant QC file provided" in caplog.text

    def test_associate_hq_intersection_and_binary_reporting(
        self, tmp_path, monkeypatch, caplog
    ):
        """--hq-samples keeps only valid phenotype ∩ HQ and logs binary summary."""
        import logging

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text(
            "sample_id\tphenotype\n"
            "S1\t1\n"
            "S2\t0\n"
            "S3\tmissing\n"
            "S4\t1\n"
        )
        hq = tmp_path / "hq_samples.tsv"
        hq.write_text("sample_id\nS1\nS3\n")

        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3]], dtype=float)
        samples = ["S1", "S2", "S3", "S4"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 1, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 1, "method": "ols"},
                ]

        def _fake_run_association_streaming(lrr_chunks, phenotype, **_kwargs):
            for _chunk, _vars in lrr_chunks:
                pass
            assert phenotype.shape == (1,)
            assert float(phenotype[0]) == 1.0
            return (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]})

        monkeypatch.setattr(association, "run_association_streaming", _fake_run_association_streaming)

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--hq-samples", str(hq),
                "--method", "ols",
                "--no-exclude-monomorphic-lrr",
                "-o", str(out),
            ])

        assert rc == 0
        assert out.exists()
        assert "Association HQ intersection" in caplog.text
        assert "dropped_lq_with_valid_pheno=2" in caplog.text
        assert "Phenotype summary (binary analyzed)" in caplog.text

    def test_associate_hq_intersection_and_quantitative_reporting(
        self, tmp_path, monkeypatch, caplog
    ):
        """Quantitative phenotype logs pre/post summaries with HQ filtering."""
        import logging

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text(
            "sample_id\tphenotype\n"
            "S1\t1\n"
            "S2\t2\n"
            "S3\t3\n"
            "S4\t4\n"
        )
        hq = tmp_path / "hq_samples.tsv"
        hq.write_text("S1\nS2\n")

        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3]], dtype=float)
        samples = ["S1", "S2", "S3", "S4"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 2, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 2, "method": "ols"},
                ]

        def _fake_run_association_streaming(lrr_chunks, phenotype, **_kwargs):
            for _chunk, _vars in lrr_chunks:
                pass
            assert phenotype.shape == (2,)
            assert np.array_equal(phenotype, np.array([1.0, 2.0]))
            return (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]})

        monkeypatch.setattr(association, "run_association_streaming", _fake_run_association_streaming)

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--hq-samples", str(hq),
                "--method", "ols",
                "-o", str(out),
            ])

        assert rc == 0
        assert out.exists()
        assert "Phenotype summary (quantitative pre-filter)" in caplog.text
        assert "n=4, mean=2.5" in caplog.text
        assert "Phenotype summary (quantitative analyzed)" in caplog.text
        assert "n=2, mean=1.5" in caplog.text

    def test_associate_hq_derived_from_sample_sheet(
        self, tmp_path, monkeypatch, caplog
    ):
        """When --hq-samples is omitted, HQ is derived from --sample-sheet."""
        import logging

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text(
            "sample_id\tphenotype\n"
            "S1\t1\n"
            "S2\t0\n"
            "S3\t1\n"
            "S4\t0\n"
        )
        # S1, S4 are HQ; S2 (low call_rate), S3 (high lrr_sd) are LQ
        sheet = tmp_path / "sheet.tsv"
        sheet.write_text(
            "Sample_ID\tcall_rate\tlrr_sd\tPC1\tPC2\n"
            "S1\t0.99\t0.10\t0.1\t0.2\n"
            "S2\t0.80\t0.10\t0.3\t0.4\n"
            "S3\t0.99\t0.50\t0.5\t0.6\n"
            "S4\t0.98\t0.34\t0.7\t0.8\n"
        )

        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3]], dtype=float,
        )
        samples = ["S1", "S2", "S3", "S4"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 2, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 2, "method": "ols"},
                ]

        def _fake_run_association_streaming(lrr_chunks, phenotype, **_kwargs):
            # S1 and S4 are HQ with valid phenotypes → 2 samples
            for _chunk, _vars in lrr_chunks:
                pass
            assert phenotype.shape == (2,)
            return (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]})

        monkeypatch.setattr(association, "run_association_streaming", _fake_run_association_streaming)

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--sample-sheet", str(sheet),
                "--method", "ols",
                "-o", str(out),
            ])

        assert rc == 0
        assert out.exists()
        assert "Deriving HQ samples from sample sheet" in caplog.text
        assert "Association HQ intersection (source=sample_sheet)" in caplog.text
        assert "dropped_lq_with_valid_pheno=2" in caplog.text

    def test_associate_hq_derived_custom_thresholds(
        self, tmp_path, monkeypatch, caplog
    ):
        """CLI --max-lrr-sd / --min-sample-call-rate override config defaults."""
        import logging

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text(
            "sample_id\tphenotype\n"
            "S1\t1\n"
            "S2\t0\n"
        )
        # S1 passes strict thresholds; S2 fails the stricter max_lrr_sd=0.05
        sheet = tmp_path / "sheet.tsv"
        sheet.write_text(
            "Sample_ID\tcall_rate\tlrr_sd\tPC1\n"
            "S1\t0.99\t0.04\t0.1\n"
            "S2\t0.99\t0.10\t0.2\n"
        )

        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2], [0.0, 0.1]], dtype=float)
        samples = ["S1", "S2"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 1, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 1, "method": "ols"},
                ]

        def _fake_run_association_streaming(lrr_chunks, phenotype, **_kwargs):
            for _chunk, _vars in lrr_chunks:
                pass
            assert phenotype.shape == (1,)
            return (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]})

        monkeypatch.setattr(association, "run_association_streaming", _fake_run_association_streaming)

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--sample-sheet", str(sheet),
                "--max-lrr-sd", "0.05",
                "--method", "ols",
                "--no-exclude-monomorphic-lrr",
                "-o", str(out),
            ])

        assert rc == 0
        assert "max_lrr_sd=0.0500" in caplog.text
        assert "dropped_lq_with_valid_pheno=1" in caplog.text

    def test_associate_logs_ld_prune_disabled(
        self, tmp_path, monkeypatch, caplog
    ):
        """When --no-ld-prune is set, a log message is emitted."""
        import logging

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        dosage = np.array(
            [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=float,
        )
        gt_variants = [
            {"chrom": "chr1", "pos": 100, "id": "v1", "ref": "A", "alts": ("C",)},
            {"chrom": "chr1", "pos": 120, "id": "v3", "ref": "A", "alts": ("T",)},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.genotypes.read_genotypes",
            lambda *_a, **_k: (dosage.copy(), samples, list(gt_variants)),
        )
        monkeypatch.setattr(
            "array_lrr_gwas.grm.compute_grm",
            lambda d, **_k: np.eye(d.shape[1], dtype=float),
        )

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "lmm"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "lmm"},
                ]

        monkeypatch.setattr(
            association, "run_association_streaming",
            lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}),
        )

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--method", "lmm",
                "--no-ld-prune",
                "-o", str(out),
            ])

        assert rc == 0
        assert "LD pruning disabled" in caplog.text

    def test_associate_hq_samples_file_in_audit_trail(
        self, tmp_path, monkeypatch, caplog
    ):
        """--hq-samples path records association_sample_qc in the audit trail."""
        import logging

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text(
            "sample_id\tphenotype\n"
            "S1\t1.0\nS2\t2.0\nS3\t3.0\nS4\t4.0\n"
        )
        hq = tmp_path / "hq.tsv"
        hq.write_text("S1\nS2\n")

        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")
        audit_dir = tmp_path / "audit"

        lrr = np.array([[0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3]], dtype=float)
        samples = ["S1", "S2", "S3", "S4"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, assoc_variants)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([0.5, 0.3])
            stat = np.array([0.6, 0.9])
            beta = np.array([0.1, 0.2])
            se = np.array([0.15, 0.2])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.1, "se": 0.15, "stat": 0.6, "p_value": 0.5,
                     "n_samples": 2, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.2, "se": 0.2, "stat": 0.9, "p_value": 0.3,
                     "n_samples": 2, "method": "ols"},
                ]

        monkeypatch.setattr(association, "run_association_streaming",
                            lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}))

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            with caplog.at_level(logging.INFO, logger="array_lrr_gwas.audit"):
                rc = main([
                    "associate", str(fake_bcf),
                    "--phenotype", str(pheno),
                    "--hq-samples", str(hq),
                    "--method", "ols",
                    "--no-exclude-monomorphic-lrr",
                    "--audit-dir", str(audit_dir),
                    "-o", str(out),
                ])

        assert rc == 0
        # Audit log should record the sample QC stage
        assert "Audit [association_sample_qc]" in caplog.text
        # Audit files should be written
        assert (audit_dir / "associate_audit.tsv").exists()
        audit_text = (audit_dir / "associate_audit.tsv").read_text()
        assert "association_sample_qc" in audit_text
        # S3 and S4 should appear as excluded
        assert "S3" in audit_text
        assert "S4" in audit_text

    def test_associate_grm_audit_trail(
        self, tmp_path, monkeypatch, caplog
    ):
        """LMM run records grm_ld_prune and grm_samples in the audit trail."""
        import logging

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")
        audit_dir = tmp_path / "audit"

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples_list = ["S1", "S2", "S3"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples_list, assoc_variants)

        dosage = np.array(
            [[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 1.0, 0.0]], dtype=float
        )
        gt_variants = [
            {"chrom": "chr1", "pos": 100, "id": "v1", "ref": "A", "alts": ("C",)},
            {"chrom": "chr1", "pos": 110, "id": "v2", "ref": "A", "alts": ("G",)},
            {"chrom": "chr1", "pos": 120, "id": "v3", "ref": "A", "alts": ("T",)},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.genotypes.read_genotypes",
            lambda *_a, **_k: (dosage.copy(), list(samples_list), list(gt_variants)),
        )
        # LD pruning keeps v1 and v3, removes v2
        monkeypatch.setattr(
            "array_lrr_gwas.ld_prune.ld_prune",
            lambda *_a, **_k: np.array([True, False, True], dtype=bool),
        )
        monkeypatch.setattr(
            "array_lrr_gwas.grm.compute_grm",
            lambda d, **_k: np.eye(d.shape[1], dtype=float),
        )

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([0.5, 0.3])
            stat = np.array([0.6, 0.9])
            beta = np.array([0.1, 0.2])
            se = np.array([0.15, 0.2])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.1, "se": 0.15, "stat": 0.6, "p_value": 0.5,
                     "n_samples": 3, "method": "lmm"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.2, "se": 0.2, "stat": 0.9, "p_value": 0.3,
                     "n_samples": 3, "method": "lmm"},
                ]

        monkeypatch.setattr(association, "run_association_streaming",
                            lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}))

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            with caplog.at_level(logging.INFO, logger="array_lrr_gwas.audit"):
                rc = main([
                    "associate", str(fake_bcf),
                    "--phenotype", str(pheno),
                    "--method", "lmm",
                    "--ld-backend", "numpy",
                    "--audit-dir", str(audit_dir),
                    "-o", str(out),
                ])

        assert rc == 0
        assert "Audit [grm_ld_prune]" in caplog.text
        assert "Audit [grm_samples]" in caplog.text
        # Audit file should contain both stages
        audit_text = (audit_dir / "associate_audit.tsv").read_text()
        assert "grm_ld_prune" in audit_text
        assert "grm_samples" in audit_text
        # v2 should be in the audit trail as excluded (ld_prune)
        assert "v2" in audit_text

    def test_associate_result_summary_logged(
        self, tmp_path, monkeypatch, caplog
    ):
        """Result summary stats (min_p, lambda_gc, n_gws) are logged."""
        import logging

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples_list = ["S1", "S2", "S3"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples_list, assoc_variants)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([1e-10, 0.3])
            stat = np.array([5.0, 1.0])
            beta = np.array([0.5, 0.1])
            se = np.array([0.1, 0.1])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.5, "se": 0.1, "stat": 5.0, "p_value": 1e-10,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.1, "se": 0.1, "stat": 1.0, "p_value": 0.3,
                     "n_samples": 3, "method": "ols"},
                ]

        monkeypatch.setattr(association, "run_association_streaming",
                            lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}))

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate", str(fake_bcf),
                "--phenotype", str(pheno),
                "--method", "ols",
                "--no-exclude-monomorphic-lrr",
                "-o", str(out),
            ])

        assert rc == 0
        assert "Result summary:" in caplog.text
        assert "min_p=" in caplog.text
        assert "lambda_gc=" in caplog.text
        assert "n_genome_wide_sig=" in caplog.text
        assert "Effect size summary:" in caplog.text

    def test_associate_scan_config_logged(
        self, tmp_path, monkeypatch, caplog
    ):
        """Pre-scan log includes n_markers, n_samples, n_covariates."""
        import logging

        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples_list = ["S1", "S2", "S3"]
        assoc_variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples_list, assoc_variants)

        class _FakeResult:
            variant_id = ['a1', 'a2']
            chrom = ["chr1", "chr1"]
            p_value = np.array([0.5, 0.3])
            stat = np.array([0.6, 0.9])
            beta = np.array([0.1, 0.2])
            se = np.array([0.15, 0.2])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.1, "se": 0.15, "stat": 0.6, "p_value": 0.5,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.2, "se": 0.2, "stat": 0.9, "p_value": 0.3,
                     "n_samples": 3, "method": "ols"},
                ]

        monkeypatch.setattr(association, "run_association_streaming",
                            lambda *_a, **_k: (_FakeResult(), {"n_total": 2, "n_tested": 2, "n_intensity_only": 0, "n_monomorphic": 0, "excluded_markers": {}, "tested_mono_flags": [False, False]}))

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate", str(fake_bcf),
                "--phenotype", str(pheno),
                "--method", "ols",
                "--no-exclude-monomorphic-lrr",
                "-o", str(out),
            ])

        assert rc == 0
        assert "n_markers_eligible=" in caplog.text
        assert "n_samples=" in caplog.text
        assert "n_covariates=" in caplog.text

    def test_segment_help_flag(self):
        with pytest.raises(SystemExit) as exc:
            main(["segment", "--help"])
        assert exc.value.code == 0

    def test_segment_missing_input(self, tmp_path):
        out = tmp_path / "regions.bed"
        rc = main([
            "segment",
            str(tmp_path / "nonexistent.tsv"),
            "-o", str(out),
        ])
        assert rc == 1

    def test_segment_threshold(self, tmp_path):
        """Threshold segmentation via CLI with synthetic association TSV."""
        import csv

        tsv = tmp_path / "results.tsv"
        rows = [
            {"chrom": "chr1", "pos": 1000, "variant_id": "v0",
             "beta": 0.5, "se": 0.05, "stat": 10.0,
             "p_value": 1e-15, "n_samples": 100, "method": "ols"},
            {"chrom": "chr1", "pos": 2000, "variant_id": "v1",
             "beta": 0.01, "se": 0.05, "stat": 0.2,
             "p_value": 0.84, "n_samples": 100, "method": "ols"},
        ]
        with open(tsv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                                    delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

        out = tmp_path / "regions.bed"
        rc = main([
            "segment",
            str(tsv),
            "--strategy", "threshold",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert lines[0].startswith("#")
        assert len(lines) == 2  # header + 1 region

    def test_segment_hmm(self, tmp_path):
        """HMM segmentation via CLI with synthetic association TSV."""
        import csv

        tsv = tmp_path / "results.tsv"
        rows = []
        for i in range(20):
            p = 1e-12 if 5 <= i <= 10 else 0.5
            rows.append({
                "chrom": "chr1", "pos": 1000 + i * 100,
                "variant_id": f"v{i}", "beta": 0.3 if p < 0.01 else 0.01,
                "se": 0.05, "stat": 6.0 if p < 0.01 else 0.2,
                "p_value": p, "n_samples": 100, "method": "ols",
            })
        with open(tsv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                                    delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

        out = tmp_path / "regions.bed"
        rc = main([
            "segment",
            str(tsv),
            "--strategy", "hmm",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert lines[0].startswith("#")
        assert len(lines) >= 2  # header + at least 1 region
