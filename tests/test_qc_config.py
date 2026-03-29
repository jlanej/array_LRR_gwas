"""Tests for YAML-based QC configuration."""

from __future__ import annotations

import pytest

from array_lrr_gwas.qc_config import defaults, load_config, apply_to_correct_args


class TestDefaults:
    def test_returns_dict(self):
        cfg = defaults()
        assert isinstance(cfg, dict)

    def test_has_required_sections(self):
        cfg = defaults()
        assert "sample_qc" in cfg
        assert "marker_qc" in cfg
        assert "correction" in cfg

    def test_default_values(self):
        cfg = defaults()
        assert cfg["sample_qc"]["max_lrr_sd"] == 0.35
        assert cfg["sample_qc"]["min_call_rate"] == 0.97
        assert cfg["marker_qc"]["min_call_rate"] == 0.95
        assert cfg["marker_qc"]["min_var"] == 0.001
        assert cfg["marker_qc"]["max_var"] is None
        assert cfg["correction"]["k"] is None
        assert cfg["correction"]["backend"] == "rsvd"

    def test_deep_copy(self):
        """Mutating the returned dict does not affect future calls."""
        cfg1 = defaults()
        cfg1["sample_qc"]["max_lrr_sd"] = 999
        cfg2 = defaults()
        assert cfg2["sample_qc"]["max_lrr_sd"] == 0.35


class TestLoadConfig:
    def test_partial_override(self, tmp_path):
        cfg_file = tmp_path / "qc.yaml"
        cfg_file.write_text(
            "sample_qc:\n"
            "  max_lrr_sd: 0.25\n"
        )
        cfg = load_config(cfg_file)
        # Overridden
        assert cfg["sample_qc"]["max_lrr_sd"] == 0.25
        # Defaults preserved
        assert cfg["sample_qc"]["min_call_rate"] == 0.97
        assert cfg["marker_qc"]["min_call_rate"] == 0.95

    def test_full_override(self, tmp_path):
        cfg_file = tmp_path / "qc.yaml"
        cfg_file.write_text(
            "sample_qc:\n"
            "  max_lrr_sd: 0.20\n"
            "  min_call_rate: 0.99\n"
            "marker_qc:\n"
            "  min_call_rate: 0.98\n"
            "  min_var: 0.005\n"
            "  max_var: 10.0\n"
            "correction:\n"
            "  k: 7\n"
            "  backend: fbpca\n"
        )
        cfg = load_config(cfg_file)
        assert cfg["sample_qc"]["max_lrr_sd"] == 0.20
        assert cfg["sample_qc"]["min_call_rate"] == 0.99
        assert cfg["marker_qc"]["min_call_rate"] == 0.98
        assert cfg["marker_qc"]["min_var"] == 0.005
        assert cfg["marker_qc"]["max_var"] == 10.0
        assert cfg["correction"]["k"] == 7
        assert cfg["correction"]["backend"] == "fbpca"

    def test_empty_file_returns_defaults(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = load_config(cfg_file)
        assert cfg == defaults()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_unknown_section_raises(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("unknown_section:\n  foo: bar\n")
        with pytest.raises(ValueError, match="Unrecognised config sections"):
            load_config(cfg_file)

    def test_non_dict_section_raises(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("sample_qc: 42\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_config(cfg_file)

    def test_non_dict_top_level_raises(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_config(cfg_file)


class TestApplyToCorrectArgs:
    def test_defaults_mapping(self):
        cfg = defaults()
        args = apply_to_correct_args(cfg)
        assert args["max_lrr_sd"] == 0.35
        assert args["min_sample_call_rate"] == 0.97
        assert args["min_marker_call_rate"] == 0.95
        assert args["min_var"] == 0.001
        assert args["max_var"] is None
        assert args["k"] is None
        assert args["backend"] == "rsvd"

    def test_cli_overrides_take_precedence(self):
        cfg = defaults()
        cli = {"max_lrr_sd": 0.20, "k": 3}
        args = apply_to_correct_args(cfg, cli)
        assert args["max_lrr_sd"] == 0.20
        assert args["k"] == 3
        # Non-overridden values stay at config defaults
        assert args["min_sample_call_rate"] == 0.97

    def test_none_cli_overrides_ignored(self):
        cfg = defaults()
        cli = {"max_lrr_sd": None, "k": None}
        args = apply_to_correct_args(cfg, cli)
        # None values don't override config
        assert args["max_lrr_sd"] == 0.35
        assert args["k"] is None

    def test_config_plus_cli(self, tmp_path):
        cfg_file = tmp_path / "qc.yaml"
        cfg_file.write_text(
            "sample_qc:\n"
            "  max_lrr_sd: 0.30\n"
            "correction:\n"
            "  k: 5\n"
        )
        cfg = load_config(cfg_file)
        # CLI overrides k but not max_lrr_sd
        cli = {"k": 10}
        args = apply_to_correct_args(cfg, cli)
        assert args["max_lrr_sd"] == 0.30  # from YAML
        assert args["k"] == 10  # from CLI
