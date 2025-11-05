# Vehicle Classification - Modulares Python-Projekt

Ich erstelle ein modulares und generisches Python-Projekt für die Fahrzeugklassifizierung basierend auf dem bereitgestellten Jupyter Notebook.

## Projektstruktur

```
vehicle_classification/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── validator.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineer.py
│   │   └── selector.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── classifier.py
│   │   └── trainer.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── eda.py
│   │   └── results.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_features.py
├── notebooks/
│   └── exploration.ipynb
├── config/
│   └── params.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
├── requirements.txt
├── setup.py
├── main.py
└── README.md
```

## Implementierung

### 1. Hauptmodul (`main.py`)

```python
#!/usr/bin/env python3
"""
Hauptmodul für die Fahrzeugklassifizierung
"""

import logging
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.visualization.eda import EDAVisualizer
from src.visualization.results import ResultsVisualizer
from src.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Hauptpipeline für die Fahrzeugklassifizierung"""

    # Konfiguration laden
    config = Config()

    try:
        # 1. Daten laden
        logger.info("Lade Daten...")
        loader = DataLoader(config)
        df = loader.load_data()

        # 2. Explorative Datenanalyse
        logger.info("Führe explorative Datenanalyse durch...")
        eda = EDAVisualizer(config)
        eda.perform_eda(df)

        # 3. Daten vorverarbeiten
        logger.info("Verarbeite Daten...")
        preprocessor = DataPreprocessor(config)
        df_clean = preprocessor.preprocess_data(df)

        # 4. Feature-Engineering
        logger.info("Erstelle Features...")
        feature_engineer = FeatureEngineer(config)
        X, y, feature_names = feature_engineer.create_features(df_clean)

        # 5. Modelle trainieren
        logger.info("Trainiere Modelle...")
        trainer = ModelTrainer(config)
        results = trainer.train_and_evaluate(X, y)

        # 6. Ergebnisse visualisieren
        logger.info("Visualisiere Ergebnisse...")
        results_viz = ResultsVisualizer(config)
        results_viz.plot_results(results)

        logger.info("Pipeline erfolgreich abgeschlossen!")

    except Exception as e:
        logger.error(f"Fehler in der Pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
```

### 2. Konfigurationsmodul (`src/utils/config.py`)

```python
import yaml
import os
from typing import Dict, Any

class Config:
    """Zentrale Konfigurationsverwaltung"""

    def __init__(self, config_path: str = "config/params.yaml"):
        self.config_path = config_path
        self.params = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Lade Konfiguration aus YAML-Datei"""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key: str, default=None):
        """Hole Konfigurationswert"""
        return self.params.get(key, default)

    def update(self, updates: Dict[str, Any]):
        """Aktualisiere Konfiguration"""
        self.params.update(updates)
```

### 3. Datenladungsmodul (`src/data/loader.py`)

```python
import pandas as pd
import os
from typing import Optional
from src.utils.config import Config

class DataLoader:
    """Generischer Datenlader für verschiedene Quellen"""

    def __init__(self, config: Config):
        self.config = config
        self.data_path = config.get('data_path', 'data/raw/vehicle.csv')

    def load_data(self) -> pd.DataFrame:
        """Lade Daten basierend auf Konfiguration"""
        load_method = self.config.get('load_method', 'local')

        if load_method == 'local':
            return self._load_local()
        elif load_method == 'url':
            return self._load_from_url()
        else:
            raise ValueError(f"Unbekannte Lademethode: {load_method}")

    def _load_local(self) -> pd.DataFrame:
        """Lade Daten von lokaler Datei"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {self.data_path}")

        return pd.read_csv(self.data_path)

    def _load_from_url(self) -> pd.DataFrame:
        """Lade Daten von URL"""
        url = self.config.get('data_url')
        if not url:
            raise ValueError("URL muss in Konfiguration angegeben werden")

        return pd.read_csv(url)
```

### 4. Datenvorverarbeitungsmodul (`src/data/preprocessor.py`)

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from src.utils.config import Config

class DataPreprocessor:
    """Generischer Datenvorverarbeiter"""

    def __init__(self, config: Config):
        self.config = config
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Führe vollständige Datenvorverarbeitung durch"""

        # 1. Datenvalidierung
        df_valid = self._validate_data(df)

        # 2. Fehlende Werte behandeln
        df_imputed = self._handle_missing_values(df_valid)

        # 3. Ausreißer behandeln
        df_clean = self._handle_outliers(df_imputed)

        # 4. Duplikate entfernen
        df_final = self._remove_duplicates(df_clean)

        return df_final

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validiere Datenqualität"""
        # Implementiere Validierungslogik
        if df.empty:
            raise ValueError("DataFrame ist leer")

        required_columns = self.config.get('required_columns', [])
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Erforderliche Spalte fehlt: {col}")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Behandle fehlende Werte"""
        missing_strategy = self.config.get('missing_values_strategy', 'drop')

        if missing_strategy == 'drop':
            return df.dropna()
        elif missing_strategy == 'impute':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            return df
        else:
            raise ValueError(f"Unbekannte Strategie: {missing_strategy}")

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Behandle Ausreißer mit IQR-Methode"""
        if not self.config.get('handle_outliers', True):
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap outliers
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Entferne Duplikate"""
        return df.drop_duplicates().reset_index(drop=True)
```

### 5. Feature-Engineering-Modul (`src/features/engineer.py`)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from src.utils.config import Config

class FeatureEngineer:
    """Feature-Engineering für generische Datensätze"""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.label_encoder = LabelEncoder()

    def create_features(self, df: pd.DataFrame):
        """Erstelle Features für Modelltraining"""

        # Zielvariable identifizieren
        target_column = self.config.get('target_column', 'class')
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode Zielvariable
        y_encoded = self.label_encoder.fit_transform(y)

        # Feature-Transformation
        X_processed = self._transform_features(X)

        # Feature-Auswahl
        X_selected = self._select_features(X_processed)

        return X_selected, y_encoded, self.label_encoder.classes_

    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transformiere Features"""

        # Numerische Features skalieren
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

        # Kategorische Features encoden
        categorical_cols = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        return X

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Führe Feature-Auswahl durch"""
        selection_method = self.config.get('feature_selection', 'all')

        if selection_method == 'pca':
            n_components = self.config.get('pca_components', 0.95)
            self.pca.n_components = n_components
            return pd.DataFrame(self.pca.fit_transform(X))

        elif selection_method == 'variance':
            # Entferne Features mit niedriger Varianz
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            return pd.DataFrame(selector.fit_transform(X))

        else:  # 'all'
            return X
```

### 6. Modell-Training-Modul (`src/models/trainer.py`)

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from src.utils.config import Config

class ModelTrainer:
    """Generischer Modell-Trainer für Klassifizierungsaufgaben"""

    def __init__(self, config: Config):
        self.config = config
        self.models = self._initialize_models()

    def _initialize_models(self):
        """Initialisiere Modelle basierend auf Konfiguration"""
        models_config = self.config.get('models', {})

        models = {}
        if models_config.get('random_forest', True):
            models['Random Forest'] = RandomForestClassifier()

        if models_config.get('gradient_boosting', True):
            models['Gradient Boosting'] = GradientBoostingClassifier()

        if models_config.get('svm', True):
            models['SVM'] = SVC(probability=True)

        if models_config.get('logistic_regression', True):
            models['Logistic Regression'] = LogisticRegression()

        return models

    def train_and_evaluate(self, X, y):
        """Trainiere und evaluiere alle Modelle"""

        # Train-Test-Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42),
            stratify=y
        )

        results = {}

        for model_name, model in self.models.items():
            print(f"Training {model_name}...")

            # Hyperparameter-Tuning
            tuned_model = self._tune_hyperparameters(model, model_name, X_train, y_train)

            # Modell training
            tuned_model.fit(X_train, y_train)

            # Vorhersagen
            y_pred = tuned_model.predict(X_test)
            y_pred_proba = tuned_model.predict_proba(X_test) if hasattr(tuned_model, 'predict_proba') else None

            # Evaluation
            results[model_name] = self._evaluate_model(
                y_test, y_pred, y_pred_proba, tuned_model, X_train, y_train
            )

        return results

    def _tune_hyperparameters(self, model, model_name, X_train, y_train):
        """Führe Hyperparameter-Tuning durch"""

        param_grids = self.config.get('hyperparameters', {})

        if model_name in param_grids:
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=self.config.get('cv_folds', 5),
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_

        return model

    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model, X_train, y_train):
        """Evaluieren Sie das Modell mit verschiedenen Metriken"""

        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'model': model,
            'predictions': y_pred,
            'true_values': y_true
        }

        if y_pred_proba is not None:
            results['roc_auc'] = roc_auc_score(
                y_true, y_pred_proba, multi_class='ovo', average='weighted'
            )

        # Kreuzvalidierung
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.config.get('cv_folds', 5),
            scoring='accuracy'
        )
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()

        return results
```

### 7. Konfigurationsdatei (`config/params.yaml`)

```yaml
# Datenkonfiguration
data_path: "data/raw/vehicle.csv"
load_method: "local"
target_column: "class"
test_size: 0.2
random_state: 42

# Datenvorverarbeitung
missing_values_strategy: "drop"
handle_outliers: true

# Feature-Engineering
feature_selection: "all"  # all, pca, variance
pca_components: 0.95

# Modelle
models:
  random_forest: true
  gradient_boosting: true
  svm: true
  logistic_regression: true

# Hyperparameter
hyperparameters:
  Random Forest:
    n_estimators: [100, 200]
    max_depth: [10, 20, None]
    min_samples_split: [2, 5]
  Gradient Boosting:
    n_estimators: [100, 200]
    learning_rate: [0.1, 0.05]
    max_depth: [3, 5]
  SVM:
    C: [0.1, 1, 10]
    kernel: ['linear', 'rbf']
  Logistic Regression:
    C: [0.1, 1, 10]
    penalty: ['l1', 'l2']

# Cross-Validation
cv_folds: 5

# Visualisierung
plot_style: "seaborn"
color_palette: "viridis"
```

### 8. Requirements (`requirements.txt`)

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
PyYAML>=6.0
jupyter>=1.0.0
ipywidgets>=7.6.0
```

### 9. Setup-Datei (`setup.py`)

```python
from setuptools import setup, find_packages

setup(
    name="vehicle_classification",
    version="1.0.0",
    description="Modulares System für Fahrzeugklassifizierung",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.8",
)
```

## Verwendung

1. **Projekt einrichten**:
```bash
pip install -r requirements.txt
python setup.py develop
```

2. **Daten vorbereiten**:
```bash
mkdir -p data/raw
# vehicle.csv in data/raw/ kopieren
```

3. **Pipeline ausführen**:
```bash
python main.py
```

## Hauptvorteile dieses Designs

1. **Modularität**: Jede Komponente ist unabhängig und wiederverwendbar
2. **Generizität**: Funktioniert mit verschiedenen Datensätzen durch Konfiguration
3. **Erweiterbarkeit**: Einfach neue Modelle/Features hinzufügbar
4. **Reproduzierbarkeit**: Vollständige Konfiguration in YAML-Dateien
5. **Testbarkeit**: Separate Testmodule für jede Komponente
6. **Dokumentation**: Klare Struktur und Typ-Hinweise

Dieses Projekt bietet eine solide Grundlage für die Fahrzeugklassifizierung und kann leicht an andere Klassifizierungsprobleme angepasst werden.