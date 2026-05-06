"""
paths.py

Zentrale Pfad-Konfiguration für das gesamte Projekt.
Alle anderen Module importieren Pfade von hier statt sie lokal neu zu definieren.
"""

from pathlib import Path

# Projekt-Wurzel (src/Favorita_TSA/utils/paths.py → 3 Ebenen hoch)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
PREPROCESSED_DIR = PROCESSED_DIR / "preprocessed"
METRICS_DIR = DATA_ROOT / "metrics"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
IMG_DIR = PROJECT_ROOT / "img" / "mlflow"
