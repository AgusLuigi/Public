"""
test_imports.py

Smoke-Test: Stellt sicher dass alle Module importierbar sind.
Fängt fehlende Imports (z.B. 'Path not defined') sofort ab.
"""


def test_import_data_preparation():
    from Favorita_TSA.models import data_preparation  # noqa: F401


def test_import_baseline():
    from Favorita_TSA.models import baseline  # noqa: F401


def test_import_forecastability():
    from Favorita_TSA.utils import forecastability  # noqa: F401


def test_import_preprocess_data():
    from Favorita_TSA.utils import preprocess_data  # noqa: F401


def test_import_data_loader():
    from Favorita_TSA.utils import data_loader  # noqa: F401


def test_import_date_utils():
    from Favorita_TSA.utils import date_utils  # noqa: F401


def test_import_config():
    from Favorita_TSA.utils import config  # noqa: F401


def test_import_paths():
    from Favorita_TSA.utils import paths  # noqa: F401


def test_mlflow_tracking_uri_valid():
    """MLflow muss set_tracking_uri + set_experiment ohne Exception akzeptieren."""
    import mlflow

    from Favorita_TSA.utils.paths import MLRUNS_DIR

    uri = MLRUNS_DIR.as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test_smoke")  # testet die vollständige MLflow-Kette
    assert mlflow.get_tracking_uri() == uri
