"""
test_paths.py

Tests für src/Favorita_TSA/utils/paths.py
Stellt sicher dass alle Pfad-Konstanten korrekt auf den Projektstamm zeigen.
"""

from Favorita_TSA.utils.paths import (
    DATA_ROOT,
    IMG_DIR,
    METRICS_DIR,
    MLRUNS_DIR,
    PREPROCESSED_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
)


class TestProjectRoot:
    def test_is_absolute(self):
        assert PROJECT_ROOT.is_absolute()

    def test_points_to_repo_root(self):
        # pyproject.toml liegt im Projektstamm
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_src_dir_exists(self):
        assert (PROJECT_ROOT / "src").exists()

    def test_configs_dir_exists(self):
        assert (PROJECT_ROOT / "configs").exists()


class TestDataPaths:
    def test_data_root_is_child_of_project(self):
        assert DATA_ROOT == PROJECT_ROOT / "data"

    def test_raw_dir(self):
        assert RAW_DIR == DATA_ROOT / "raw"

    def test_processed_dir(self):
        assert PROCESSED_DIR == DATA_ROOT / "processed"

    def test_preprocessed_dir(self):
        assert PREPROCESSED_DIR == PROCESSED_DIR / "preprocessed"

    def test_metrics_dir(self):
        assert METRICS_DIR == DATA_ROOT / "metrics"


class TestAppPaths:
    def test_mlruns_dir(self):
        assert MLRUNS_DIR == PROJECT_ROOT / "mlruns"

    def test_img_dir(self):
        assert IMG_DIR == PROJECT_ROOT / "img" / "mlflow"
