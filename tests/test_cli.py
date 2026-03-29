"""Tests for the CLI entry point."""

from pathlib import Path

import numpy as np
import pytest

from array_lrr_gwas.cli import _build_parser, _variant_id, main


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

    def test_correct_no_build_detected(self, test_bcf_path, tmp_path):
        """When build is not detectable and not supplied, exits with error."""
        out = tmp_path / "out.bcf"
        rc = main(["correct", str(test_bcf_path), "-o", str(out)])
        # Our test BCF has no contig lengths, so build detection fails
        assert rc == 1

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

    def test_associate_lmm_no_gt_fails(self, test_bcf_path, tmp_path):
        """LMM with a BCF that has no GT data should fail gracefully."""
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
            "-o", str(out),
        ])
        assert rc == 1  # Should fail: no GT data available

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
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, assoc_variants),
        )

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
            chrom = ["chr1", "chr1"]

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
            "run_association",
            lambda *_a, **_k: _FakeResult(),
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
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, assoc_variants),
        )

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
            chrom = ["chr1", "chr1"]

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
            "run_association",
            lambda *_a, **_k: _FakeResult(),
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
        monkeypatch.setattr("array_lrr_gwas.io_vcf.read_lrr", lambda _p: (lrr, samples, assoc_variants))

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
            chrom = ["chr1"]

            @staticmethod
            def to_records():
                return [{
                    "chrom": "chr1", "pos": 100, "variant_id": "a1",
                    "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                    "n_samples": 3, "method": "lmm",
                }]

        monkeypatch.setattr(association, "run_association", lambda *_a, **_k: _FakeResult())

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
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, assoc_variants),
        )

        class _FakeResult:
            chrom = ["chr1", "chr1"]

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
            association, "run_association",
            lambda *_a, **_k: _FakeResult(),
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

        # Check values for a1
        assert rows[0]["all_ancestries_call_rate_pass"] == "True"
        assert rows[0]["all_ancestries_hwe_pass"] == "True"
        assert rows[0]["all_ancestries_maf_pass"] == "False"

        # Check values for a2
        assert rows[1]["all_ancestries_call_rate_pass"] == "False"
        assert rows[1]["all_ancestries_hwe_pass"] == "True"
        assert rows[1]["all_ancestries_maf_pass"] == "True"

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
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, assoc_variants),
        )

        class _FakeResult:
            chrom = ["chr1", "chr1"]

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
            association, "run_association",
            lambda *_a, **_k: _FakeResult(),
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
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, assoc_variants),
        )

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
            chrom = ["chr1", "chr1"]

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
            association, "run_association",
            lambda *_a, **_k: _FakeResult(),
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
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, assoc_variants),
        )

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
            chrom = ["chr1", "chr1"]

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
            association, "run_association",
            lambda *_a, **_k: _FakeResult(),
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
