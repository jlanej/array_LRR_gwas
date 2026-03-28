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
