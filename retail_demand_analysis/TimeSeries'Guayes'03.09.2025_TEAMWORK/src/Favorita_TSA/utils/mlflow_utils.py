"""
mlflow_utils.py

Portables MLflow-Setup für das Favorita-Projekt.
setup_mlflow() passt gespeicherte Artifact-Pfade automatisch an die
aktuelle Maschine an — funktioniert für jeden Nutzer ohne manuelle Anpassung.
"""

from __future__ import annotations

import mlflow
import yaml

from Favorita_TSA.utils.paths import MLRUNS_DIR


def setup_mlflow(experiment_name: str) -> mlflow.entities.Experiment:
    """Setzt Tracking-URI, repariert Artifact-Pfade und gibt das Experiment zurück.

    Kann von jedem Nutzer auf jeder Maschine aufgerufen werden — die in
    meta.yaml gespeicherten Pfade werden automatisch auf MLRUNS_DIR angepasst.
    """
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    _fix_artifact_locations()
    return mlflow.set_experiment(experiment_name)


def _fix_artifact_locations() -> None:
    """Ersetzt fremde artifact_location-Pfade in allen Experiment-meta.yaml."""
    if not MLRUNS_DIR.exists():
        return

    local_prefix = MLRUNS_DIR.as_uri()

    for meta_file in MLRUNS_DIR.rglob("meta.yaml"):
        try:
            with open(meta_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            continue

        changed = False

        for key in ("artifact_location", "artifact_uri"):
            val = data.get(key, "")
            if val and not val.startswith(local_prefix):
                # Pfad-Anteil nach /mlruns/ extrahieren und neu zusammensetzen
                marker = "/mlruns/"
                idx = val.find(marker)
                if idx != -1:
                    suffix = val[idx + len(marker) :]
                    data[key] = f"{local_prefix}/{suffix}"
                    changed = True

        if changed:
            with open(meta_file, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
