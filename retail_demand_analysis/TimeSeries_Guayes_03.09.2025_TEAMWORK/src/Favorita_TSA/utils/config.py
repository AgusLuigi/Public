"""
config.py

Lädt configs/config.yaml und stellt die Werte als einfach zugängliches
Objekt bereit.

Verwendung:
    from Favorita_TSA.utils.config import cfg

    threshold = cfg.croston.adi_threshold
    experiment = cfg.mlflow.experiment
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "config.yaml"


def _to_namespace(obj: Any) -> Any:
    """Konvertiert dict rekursiv zu SimpleNamespace für Attribut-Zugriff."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(i) for i in obj]
    return obj


def _load() -> SimpleNamespace:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


cfg: SimpleNamespace = _load()
