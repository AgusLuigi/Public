# MEMORY.md — Favorita Forecasting Projekt

> Einbinden mit `@MEMORY.md` am Anfang einer neuen Konversation.
> Hält Claude auf Stand ohne alles neu erklären zu müssen.

---

## Projektübersicht

**Ziel:** End-to-end ML-Pipeline für Lebensmittelverkaufs-Forecasting (Kaggle Favorita Dataset)
**Stack:** Python 3.11, Pandas, Statsforecast, MLflow, Streamlit, Plotly, Parquet
**Struktur:**
```
src/
├── Favorita_TSA/        # Kern-Paket (utils, models, features, viz, preprocess)
└── streamlit_app/       # Web-App (app.py + 4 Pages + components/)
configs/                 # COLORS.yaml, config.yaml
data/                    # raw/ → processed/ → metrics/
mlruns/                  # MLflow Experiment Tracking
tests/
```

---

## Architektur & Konventionen

### Pfade
- **Immer** aus `Favorita_TSA.utils.paths` importieren — nie lokal neu definieren
- Wichtigste Konstanten: `PROJECT_ROOT`, `PREPROCESSED_DIR`, `METRICS_DIR`, `RAW_DIR`, `MLRUNS_DIR`

### Konfiguration
- Magic Numbers gehören in `configs/config.yaml`
- Laden via `from Favorita_TSA.utils.config import cfg`
- Zugriff: `cfg.croston.adi_threshold`, `cfg.mlflow.experiment`, etc.

### Datum/Zeit
- Alle Period-zu-Timestamp Konvertierungen über `Favorita_TSA.utils.date_utils`
- Funktionen: `normalize_time_col()`, `period_to_timestamp()`, `get_date_col()`

### Streamlit UI
- Charts: `from streamlit_app.components.charts import render_plotly` — nie direkt `st.plotly_chart()`
- Filter: `from streamlit_app.components.filters import render_pattern_filter`
- Metriken: `from streamlit_app.components.metrics_row import render_metrics_row`

### Aggregationen
- Aggregationsfunktionen in `preprocess_data.py` werden via `_make_aggregator()` Factory erzeugt
- Nie wieder einzelne Wrapper-Funktionen für `aggregate()` schreiben

---

## Erledigte Refactorings

### 2026-03-20 — Strukturelles Refactoring

| Was | Datei(en) |
|-----|-----------|
| `paths.py` neu erstellt | `src/Favorita_TSA/utils/paths.py` |
| `config.yaml` + `config.py` erstellt (60+ Magic Numbers) | `configs/config.yaml`, `src/Favorita_TSA/utils/config.py` |
| `date_utils.py` erstellt | `src/Favorita_TSA/utils/date_utils.py` |
| `components/` erstellt (charts, filters, metrics_row) | `src/streamlit_app/components/` |
| `df_to_parquet_packages` Duplikat entfernt | `data_loader.py` |
| 11 Aggregationsfunktionen → 1 `_make_aggregator()` Factory | `preprocess_data.py` |
| `render_daily_tab`/`render_weekly_tab` → `render_aggregation_tab(granularity)` | `forecastability_store_item.py` |
| `importlib.reload()` Hack entfernt | `model_training.py` |
| `project_root()` Funktion entfernt → `PREPROCESSED_DIR` | `store_item_behavior.py` |
| Alle Pfade auf `paths.py` umgestellt | `data_loader.py`, `preprocess_data.py`, `forecastability.py`, `data_preparation.py`, `model_training.py`, `store_item_behavior.py` |
| `ADI_THRESHOLD`/`CV2_THRESHOLD` → `cfg` | `forecastability.py` |
| AutoARIMA-Defaults → `cfg` | `model_training.py` |
| `PATTERN_DEFAULTS` → `cfg` | `model_training.py` |
| Period-Konvertierung → `date_utils` | `multi_stores.py` |
| Alle `st.plotly_chart()` → `render_plotly()` | alle Pages |
| Rolling-Window + Z-Score-Threshold → `cfg` | `multi_stores.py` |

---

## Offene Punkte / Nächste Schritte

- [ ] `store_item_behavior.py`: Zeitliche Konstanten (`7`, `52`) auf `cfg.time_series.*` umstellen
- [ ] `baseline.py`: `gap_threshold` + MLflow-Experiment-Name auf `cfg` umstellen
- [ ] Hardcoded Farben in `store_item_behavior.py` → `ColorManager`
- [ ] `backup/` Ordner aufräumen (veraltete Dateien)
- [ ] Daten-Lade-Pattern vereinheitlichen: alle Pages auf `@st.cache_data` + gleiche Konvention
- [ ] Holiday-Konfiguration (`holidays.py`, `holiday_parquets.py`) auf `cfg` umstellen
- [ ] `components/tabs.py` implementieren für Daily/Weekly Tab-Muster

---

## Wichtige Entscheidungen

| Entscheidung | Grund |
|-------------|-------|
| `paths.py` statt `os.chdir()` | `os.chdir()` ändert globalen State, macht Tests und Imports fehleranfällig |
| `config.yaml` + `SimpleNamespace` statt Klasse | Einfacher Attribut-Zugriff (`cfg.models.cv_folds`) ohne Boilerplate |
| `_make_aggregator()` Factory | 11 identische Wrapper-Funktionen waren reiner Boilerplate |
| `render_aggregation_tab(granularity)` | 93% identischer Code war nicht wartbar |
| `importlib.reload()` entfernt | War Development-Workaround, nie für Produktion gedacht |
| `render_plotly()` in components | Änderung an einem Ort wirkt auf alle 12+ Vorkommen |
