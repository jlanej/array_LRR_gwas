"""Tests for the CLI entry point."""

from pathlib import Path

import pytest

from array_lrr_gwas.cli import main


class TestCli:
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
