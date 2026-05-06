"""
test_config.py

Tests für src/Favorita_TSA/utils/config.py
Stellt sicher dass config.yaml korrekt geladen wird und alle Pflicht-Felder vorhanden sind.
"""

import pytest

from Favorita_TSA.utils.config import cfg


class TestConfigLoads:
    def test_cfg_is_not_none(self):
        assert cfg is not None

    def test_cfg_has_croston(self):
        assert hasattr(cfg, "croston")

    def test_cfg_has_parquet(self):
        assert hasattr(cfg, "parquet")

    def test_cfg_has_time_series(self):
        assert hasattr(cfg, "time_series")

    def test_cfg_has_models(self):
        assert hasattr(cfg, "models")

    def test_cfg_has_mlflow(self):
        assert hasattr(cfg, "mlflow")

    def test_cfg_has_holidays(self):
        assert hasattr(cfg, "holidays")

    def test_cfg_has_analysis(self):
        assert hasattr(cfg, "analysis")

    def test_cfg_has_ui(self):
        assert hasattr(cfg, "ui")

    def test_cfg_has_defaults(self):
        assert hasattr(cfg, "defaults")


class TestCrostonThresholds:
    def test_adi_threshold_type(self):
        assert isinstance(cfg.croston.adi_threshold, float)

    def test_cv2_threshold_type(self):
        assert isinstance(cfg.croston.cv2_threshold, float)

    def test_adi_threshold_value(self):
        assert cfg.croston.adi_threshold == pytest.approx(1.32)

    def test_cv2_threshold_value(self):
        assert cfg.croston.cv2_threshold == pytest.approx(0.49)


class TestTimeSeries:
    def test_daily_season_length(self):
        assert cfg.time_series.daily_season_length == 7

    def test_weekly_season_length(self):
        assert cfg.time_series.weekly_season_length == 52


class TestModels:
    def test_gap_threshold(self):
        assert cfg.models.gap_threshold == pytest.approx(0.05)

    def test_cv_folds(self):
        assert cfg.models.cv_folds == 3

    def test_autoarima_has_max_p(self):
        assert hasattr(cfg.models.autoarima, "max_p")

    def test_autoarima_grid_p_is_list(self):
        assert isinstance(cfg.models.autoarima.grid_p, list)

    def test_autoarima_grid_q_is_list(self):
        assert isinstance(cfg.models.autoarima.grid_q, list)


class TestMLflow:
    def test_experiment_is_string(self):
        assert isinstance(cfg.mlflow.experiment, str)

    def test_experiment_not_empty(self):
        assert len(cfg.mlflow.experiment) > 0


class TestAnalysis:
    def test_rolling_window(self):
        assert cfg.analysis.rolling_window == 8

    def test_rolling_min_periods(self):
        assert cfg.analysis.rolling_min_periods == 4

    def test_zscore_threshold(self):
        assert cfg.analysis.zscore_threshold == pytest.approx(3.0)


class TestPatternExamples:
    def test_daily_smooth_has_store(self):
        assert hasattr(cfg.defaults.pattern_examples.daily_smooth, "store")

    def test_daily_smooth_has_item(self):
        assert hasattr(cfg.defaults.pattern_examples.daily_smooth, "item")

    def test_all_four_patterns_present(self):
        pe = cfg.defaults.pattern_examples
        assert hasattr(pe, "daily_smooth")
        assert hasattr(pe, "daily_erratic")
        assert hasattr(pe, "weekly_smooth")
        assert hasattr(pe, "weekly_erratic")
