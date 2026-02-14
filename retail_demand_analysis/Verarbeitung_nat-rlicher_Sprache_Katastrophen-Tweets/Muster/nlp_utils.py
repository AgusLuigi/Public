"""
ZENTRALE FUNKTIONSSAMMLUNG (NLP_UTILS.PY)

# WICHTIG: Damit der Name erkannt wird, vorher einmal setzen:
Dieses Modul ist das Herzstück des NLP-Projekts. Es zentralisiert alle wiederkehrenden
Aufgaben, um Redundanz zu vermeiden und die Wartbarkeit zu erhöhen.

STRUKTUR & ORIENTIERUNG:
------------------------
1. GLOBALE KONFIGURATION:
   Hier werden Pfade, Farben, Spaltennamen und Standard-Parameter definiert.
   Ändern Sie Werte hier, um sie im gesamten Projekt anzupassen.

2. TEXT PREPROCESSING (KLASSEN & FUNKTIONEN):
   Werkzeuge zur Bereinigung von Textdaten (HTML entfernen, Lemmatisierung, etc.).
   Enthält Scikit-Learn kompatible Transformer für Pipelines.

3. FEATURE ENGINEERING (LOGIK & METRIKEN):
   Spezifische Funktionen für dieses Projekt:
   - LLM-Simulation (Erkennt starke Signalwörter)
   - Noise-Filter (Erkennt eBay, Gaming, etc.)
   - Disaster-Score (Berechnet Bedrohungsgrad)
   - Kontext-Extraktion (Trigramme um Keywords)

4. VEKTORDATENBANK & EMBEDDINGS:
   Klassen für die Arbeit mit Embeddings und Vektordatenbanken (Milvus),
   um semantische Ähnlichkeiten zu finden.

5. VISUALISIERUNG (PLOTLY):
   Umfangreiche Sammlung von Plot-Funktionen. Alle nutzen das globale Farbschema.
   - Verteilungen, Wordclouds, N-Gramme
   - Konfusionsmatrizen, ROC-Kurven, Precision-Recall
   - Feature Importance, Benchmark-Ergebnisse, Trainingsverlauf
   - NEU: Strategische Token-Analyse & V-Shape Decision Plot
   - NEU: Fehler-Analyse Scatterplot

6. MODELLIERUNG & PIPELINES:
   Die Hauptfunktionen zur Ausführung der Analysen:
   - run_analysis: Standard-Pipeline (Ensemble + Auto-Tuning + Hard Example Mining)
   - run_bert_training: Deep Learning mit Transformern
   - run_comprehensive_benchmark: Vergleich von 25+ Algorithmen
"""
import pyarrow
import sys
import io
import hashlib
import torch
import subprocess
import socket
import requests
import shutil
import webbrowser
import gc
import multiprocessing
import pandas as pd
import numpy as np
import re
import emoji
import nltk
import os
import json
import time
import psutil
import platform
import inflect
import joblib
from joblib import Parallel, delayed
from IPython.display import display
from langdetect import detect, DetectorFactory
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from collections import Counter
from contextlib import contextmanager

# Für die Übersetzung (installiere falls nötig: pip install deep_translator)
from deep_translator import GoogleTranslator
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback
from wordcloud import WordCloud
from scipy import sparse

# Plotly Imports (Interaktive Grafiken)
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scikit-Learn Core
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, recall_score,
                             precision_score, accuracy_score, roc_curve, auc, precision_recall_curve,
                             roc_auc_score, mean_squared_error)
from sklearn.utils import class_weight

# Scikit-Learn Models (Die komplette NLP-Palette)
from sklearn.linear_model import (LogisticRegression, SGDClassifier, RidgeClassifier,
                                  PassiveAggressiveClassifier, Perceptron, RidgeClassifierCV)
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

# Optionale Bibliotheken (Robustheit für andere Umgebungen)
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import catboost as cb
except ImportError:
    cb = None
try:
    from pymilvus import MilvusClient, model as milvus_model
except ImportError:
    MilvusClient = None

# TensorFlow / Keras / BERT
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import keras_nlp
from tensorflow.keras.callbacks import EarlyStopping

# ==============================================================================
# 1. GLOBALE KONFIGURATION & INITIALISIERUNG
# ==============================================================================

# 1.1 Ordnerstruktur
BASE_DIR = "Ablagerung_NLP"
MODEL_DIR = os.path.join(BASE_DIR, "model")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts_ml")
ENCLAVE_DIR = os.path.join(BASE_DIR, "enclave")
EDA_PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "eda_plots")

DIRECTORIES_TO_CREATE = [BASE_DIR, MODEL_DIR, CONFIG_DIR, ARTIFACTS_DIR, EDA_PLOTS_DIR, ENCLAVE_DIR]
for directory in DIRECTORIES_TO_CREATE:
    os.makedirs(directory, exist_ok=True)

# 1.2 Dateipfade (Dynamisch relativ zum Skript)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "Data_Set", "train.csv")
SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "submission.csv")

PLOT_COLORS_PATH = os.path.join(CONFIG_DIR, "plot_colors.json")
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "base_model_optimized.joblib")
BEST_CLASSIC_MODEL_PATH = os.path.join(MODEL_DIR, "best_classic_model.joblib")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "distilbert_classifier.keras")

# 1.3 Farben (Globales Schema - Angepasst an Vorgabe)
PLOT_COLORS = {
    "primary": "#1f77b4",  # Blau
    "secondary": "#ff7f0e",  # Gelb
    "background": "#808080",  # Grau Standard
    "text": "#FFFFFF",  # Weißer Text für Kontrast auf Grau
    "grid": "#dddddd",  # Grau Neutrales
    "success": "#2ca02c",  # Grün -> Keine Katastrophe (Positiv/Sicher)
    "error": "#d62728",  # Rot -> Katastrophe (Gefahr)
    "no_disaster": "#2ca02c",  # Grün
    "disaster": "#d62728",  # Rot
    "Ladebalken": "#ADD8E6",  # Hellblau
    # Skalen für Plotly Heatmaps & Radar-Flächen
    "scale_two": [[0, "#ffffff"], [1, "#1f77b4"]],
    "scale_Tree": [[0, "#2ca02c"], [0.5, "#dddddd"], [1, "#d62728"]]
}
with open(PLOT_COLORS_PATH, 'w') as f: json.dump(PLOT_COLORS, f, indent=4)

# CORE reservierung für Flüssiger betrieb
TOTAL_CORES = multiprocessing.cpu_count()
RESERVED_CORES = max(1, TOTAL_CORES - 1)
MODEL_DEFAULTS = {'n_jobs': RESERVED_CORES, 'random_state': 42, 'class_weight': 'balanced'}

# Plot Konfiguration
PLOT_CONFIG = {
    'width': 500,
    'height': 400,
    'template': 'plotly_dark'  # Dunkles Template passt besser zu grauem Hintergrund
}

# 1.4 Globale Variablen & Einstellungen (ZENTRALISIERUNG)

COLS = {
    'text': 'text',
    'target': 'target',
    'clean': 'cleaned_text',
    'length': 'length',
    'lemmatized': 'lemmatized_text'
}

# TQDM (Ladebalken) Style
TQDM_STYLE = {
    'colour': PLOT_COLORS['Ladebalken'],
    'bar_format': '{l_bar}{bar:20}{r_bar}'
}

# Logarithmische Zeitachse für bessere Spreizung
nltk.download(
    ['stopwords', 'punkt', 'wordnet', 'punkt_tab', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'],
    quiet=True)
lemmatizer = WordNetLemmatizer()

# 1.6 NLP Konstanten (Keywords & Listen)
NOISE_KEYWORDS = ['ebay', 'auction', 'sale', 'cree led', 'retro 5', 'fire red', 'discount', 'buy now',
                  'price', 'shipping', 'pre-order', 'brand new', 'fashion', 'bag', 'purse', 'deal of the day',
                  'retail', 'order online', 'free shipping', 'stock', 'handbag', 'leather', 'clearance',
                  'store', 'promo', 'coupon', 'best price', 'amazon', 'etsy',
                  'warcraft', 'roblox', 'video playlist', 'movie', 'special edition', 'rap battle',
                  'minecraft', 'twitch', 'stream', 'gameplay', 'subscribers', 'nintendo', 'fortnite',
                  'xbox', 'playstation', 'lyrics', 'official video', 'remix', 'playlist', 'soundcloud',
                  'itunes', 'spotify', 'vlog', 'cinema', 'theatre', 'actor', 'comedy', 'funny', 'parody',
                  'season finale', 'episode', 'level up', 'final boss',
                  'burning calories', 'on fire', 'blown away', 'drowning in', 'heart attack',
                  'love is a battlefield', 'work is killing me', 'sunk cost', 'flame war', 'brainstorm',
                  'spill the beans', 'light a fire', 'shot in the dark', 'under fire', 'backfire',
                  'icebreaker', 'the bomb', 'destroyed the stage', 'killed it', 'bombed the exam',
                  'mind blowing', 'fire emoji', 'lit', 'slaying', 'break a leg', 'heartbreaker', 'epic fail',
                  'ignition knock', 'detonation sensor', 'engine knock', 'spark plug', 'exhaust',
                  'battery charger', 'led bar', 'fog lamp', 'sensor', 'v8 engine', 'chassis',
                  'fuel injection', 'reset button', 'debug', 'software crash', 'system reboot',
                  'battery low', 'error 404', 'bug fix', 'firmware',
                  'sunshine', 'beautiful day', 'weather report', 'forecast', 'skyline', 'sunset',
                  'gardening', 'fishing', 'camping fire', 'cook fire', 'barbecue', 'bbq', 'picnic',
                  'baking', 'recipe', 'silence', 'quiet', 'peaceful', 'calm after the storm',
                  'inner peace', 'meditation', 'zen', 'chill', 'relaxing', 'serene',
                  'policy', 'terms and conditions', 'privacy', 'legal notice', 'disclaimer',
                  'update log', 'version history', 'patch notes', 'manual', 'user guide',
                  'hiring', 'resume', 'career', 'marketing strategy', 'tax return', 'customer service',
                  'nba', 'jersey', 'freestyle', 'soccer game', 'tryout', 'crossfit', 'goal', 'score', 'game']

DISASTER_KEYWORDS = ['attack', 'boom', 'pathogen', 'derailment', 'tornado', 'bombe', 'explosion', 'fire',
                     'flood', 'earthquake', 'hiroshima', 'storm', 'fatality', 'riot', 'bioterror', 'cyclone',
                     'crushed', 'disaster', 'emergency', 'bleeding', 'wildfire', 'outbreak', 'evacuate',
                     'terror', 'war', 'killed', 'death', 'nuclear', 'crash', 'electrocuted', 'casualty',
                     'injury', 'hostage', 'drought', 'assault', 'destruction', 'annihilated', 'mudslide',
                     'heat wave', 'sinking', 'damage', 'weapon', 'virus', 'collapsed', 'destroy', 'hazard',
                     'dangerous', 'blown', 'refugee', 'detonation', 'devastation', 'wreck', 'missing',
                     'kidnapped', 'threat', 'murder', 'avalanche', 'quarantined', 'flame', 'exploded',
                     'invasion', 'deluge', 'blizzard', 'drowning', 'hail', 'meltdown', 'burning', 'sunk',
                     'demolition', 'terrorist', 'homeless', 'snowstorm', 'bullet']

TIME_KEYWORDS = ['now', 'today', 'seconds', 'minutes', 'hours', 'noon', 'night',
                 'pm', 'am', 'o\'clock', 'just', 'currently', 'ago', 'morning',
                 'afternoon', 'evening', 'yesterday', 'tonight', 'immediate']

NEWS_KEYWORDS = ['breaking', 'news', 'reported', 'official', 'update', 'latest', 'did you know']

SITUATIVE_LOCATIONS = ['kitchen', 'living room', 'shop', 'store', 'street', 'road',
                       'forest', 'woods', 'bridge', 'house', 'building', 'basement',
                       'station', 'highway', 'park', 'airport', 'hospital', 'school']

RELATIVE_LOCATIONS = ['here', 'nearby', 'opposite', 'close by', 'in front of',
                      'on site', 'right there', 'across', 'around the corner',
                      'at the scene', 'local']

POI_LOCATIONS = ['hospital', 'school', 'airport', 'station', 'bridge',
                 'highway', 'power plant', 'mall', 'stadium', 'downtown']

EMOJI_REPLACEMENTS = {
    "collision": "crash", "gun": "weapon", "pistol": "weapon",
    "nauseated_face": "disgust", "face_vomiting": "disgust", "vomiting_face": "disgust",
    "sick": "disgust", "ill": "disgust", "fever": "disgust", "cold": "disgust",
    "fire_engine": "emergency", "police_car": "emergency", "fire_fighter": "emergency",
    "fire_truck": "emergency", "skull": "death", "bomb": "explosion", "fire": "fire",
    "siren": "alarm", "hospital": "medical", "ambulance": "medical", "police": "emergency",
    "emergency": "service", "boom": "explosion", "attack": "assault", "blast": "explosion",
    "bioterror": "terror", "pathogen": "virus", "crushed": "death", "fatality": "death",
    "electrocuted": "death", "reactor": "nuclear", "derailment": "crash", "riot": "attack",
    "cyclone": "storm", "annihilation": "death", "desolate": "disaster"
}

STRONG_LLM_SIGNALS = ['killed', 'suicide', 'bomb', 'spill', 'crash', 'disaster', 'dead', 'casualties']


# ==============================================================================
# 0. setup Enclave Ollama
# ==============================================================================
def storage_backend():
    """
    SYSTEM-HEBEL:
    Prüft pyarrow. Bei Neuinstallation wird der Kernel abgebrochen,
    damit beim nächsten Start die Treiber (Handshake) sauber geladen werden.
    """
    # 1. PRÜFUNG: Ist pyarrow bereits geladen und funktionsfähig?
    try:
        pd.DataFrame({'a': [1]}).to_parquet(io.BytesIO(), engine='pyarrow')
        print("✅(pyarrow) ist bereits einsatzbereit.")
        return True

    # 2. INSTALLATION: Falls es fehlt oder der Handshake fehlerhaft ist
    except (ImportError, Exception):
        print("\n📦 SYSTEM-VERVOLLSTÄNDIGUNG: 'pyarrow' wird eingerichtet...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyarrow"], check=True)
            print("\n" + "!" * 60)
            print("✅ INSTALLATION ERFOLGREICH!")
            print("🚀 DER KERNEL WIRD JETZT ABGEBROCHEN, UM DIE TREIBER ZU AKTIVIEREN.")
            print("👉 BITTE STARTEN SIE DIE ZELLE EINFACH NOCH EINMAL.")
            print("!" * 60 + "\n")
            os._exit(0)

        except Exception as e:
            print(f"❌ Kritischer Fehler bei der Installation: {e}")
            return False


def setup_enclave():
    """
    UNIVERSAL SETUP:
    Optimiert für MacBook Air (MPS) & Windows/Linux.
    1. Automatische Installation je nach OS.
    2. Apple Silicon GPU-Beschleunigung Check.
    3. Core-Reserve Standard (1 Core frei).
    """
    print("\n📦pyarrow installation überprüfung")
    storage_backend()

    print(f"\n🛠️ AUTOMATISCHES ENCLAVE-SETUP ({platform.system()} erkannt)...")
    # 1. BASIS-SOFTWARE CHECK & PLATTFORM-INSTALLATION
    if shutil.which('ollama') is None:
        system_os = platform.system()
        print(f"⚠️ Ollama wurde auf {system_os} nicht gefunden!")

        def has_internet():
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                return True
            except:
                return False

        if not has_internet():
            print("❌ Kein Internet. Installation abgebrochen.")
            return None
        try:
            if system_os == "Darwin":
                if shutil.which('brew'):
                    print("📥 Installiere Ollama via Homebrew...")
                    subprocess.run(["brew", "install", "--cask", "ollama"], check=True)
                else:
                    print("🌐 Homebrew fehlt. Öffne offiziellen Mac-Download...")
                    webbrowser.open("https://ollama.com/download/mac")
                    return None
            elif system_os == "Linux":
                print("📥 Starte Linux-Installations-Script...")
                subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            elif system_os == "Windows":
                print("🪟 Windows erkannt. Öffne Download für Ollama.exe...")
                webbrowser.open("https://ollama.com/download/windows")
                return None
        except Exception as e:
            print(f"❌ Automatische Installation fehlgeschlagen: {e}")
            return None

    # 2. HARDWARE-ERKENNUNG
    gpu_available = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    target_model = "llama3" if gpu_available else "nemotron-3-nano"

    print(f"🖥️ Hardware-Status: {'GPU/MPS BESCHLEUNIGT' if gpu_available else 'CPU MODUS'}")
    if not gpu_available:
        print("🛡️ '1 Core frei' Prinzip aktiv: Nutze kompaktes Nemotron.")
    print(f"🎯 Ziel-Modell: {target_model}")

    # 3. SERVER-START & MODELL-DOWNLOAD
    try:
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        check = subprocess.run(['ollama', 'list'], capture_output=True, text=True)

        if target_model not in check.stdout:
            print(f"📥 Paket {target_model} fehlt. Lade jetzt herunter...")
            subprocess.run(['ollama', 'pull', target_model], check=True)
            print(f"✅ Paket {target_model} erfolgreich lokal gespeichert.")
        else:
            print(f"✅ Modell {target_model} ist bereits einsatzbereit.")

    except Exception as e:
        print(f"⚠️ Fehler beim Modell-Management: {e}")
        return None

    return target_model


def LLM_enclave_call_offline(text, model_name):
    """
    STRIKTER OLLAMA-CALL: Nutzt das dynamisch gewählte Modell.
    1. Stellt eine Frage an das modell tabelle TEXT zeilen ist 0 oder 1 und gibt result aus
    """
    if not text or pd.isna(text):
        return 0

    prompt = (
        "System: Du bist ein Katastrophen-Analyst. Antworte NUR mit '1' (Ja) oder '0' (Nein).\n"
        f"Text: {text}\n"
        "Katastrophe?:"
    )

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 2, "temperature": 0.0}
            },
            timeout=30  # Dein gewünschter Timeout pro Reihe
        )
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            return 1 if '1' in result[:3] else 0
        return -1
    except Exception:
        return -1


# ==============================================================================
# 1. GLOBALE EDA vor Bereinigung
# ==============================================================================

def EDA_vor_reinigung(df: pd.DataFrame):
    """
    NLP-MANCOS-ANALYSE (IST-ZUSTAND)
    Analysiert: Spalten-Qualität, Sprachen (Count & %), Text-Statistik, Emojis, Shortforms & Anomalien.
    """
    print('Mancos in der Struktur (NaNs, Duplikate, Leerräume)')
    print('Mancos in der Sprache (Fremdsprachen, reine Emoji-Texte, zu kurze Texte)')
    print('Mancos im Inhalt (Slang/Kürzel, Stoppwort-Last, Vokabular-Dichte)')
    print('Mancos in der Verteilung (Wie unterscheiden sich Katastrophen von normalen Tweets?)')
    # Seed für stabile Ergebnisse der Spracherkennung
    DetectorFactory.seed = 42
    print(' ' * 15, '🚀 TEXT-DATEN-ANALYSE (IST-ZUSTAND)')
    total_rows = len(df)
    txt_col_name = 'text'  # Hier den Namen der Textspalte definieren

    # --- 1. 🌍 SPRACHANALYSE (MIT COUNT & %) ---
    if txt_col_name in df.columns:
        print("\n🌍 SPRACHANALYSE (Stichprobe 500 Einträge):")

        def quick_detect(x):
            try:
                # Manco-Check: Wenn kein Buchstabe da ist -> Reine Emojis/Symbole
                if not re.search(r'[a-zA-Z]', str(x)): return "Only Emojis/Symbols"
                return detect(x) if len(str(x)) > 10 else "too_short"
            except:
                return "unknown"

        sample_size = min(len(df), 500)
        sample_lang = df[txt_col_name].sample(sample_size).apply(quick_detect)

        # Tabelle mit absoluten Zahlen (Count) und relativen Anteilen (%)
        lang_df = pd.DataFrame({
            'Anzahl (Count)': sample_lang.value_counts(),
            'Anteil (%)': (sample_lang.value_counts(normalize=True) * 100).round(2)
        })
        display(lang_df.head(10))
    print('=' * 80)

    # 2. 📊 ÜBERSICHT DER SPALTEN-QUALITÄT (INKL. EMOJIS & KÜRZEL)
    analysis_data = []
    for col in df.columns:
        series = df[col]
        nan_count = series.isnull().sum()
        unique_count = series.nunique()
        cardinality = (unique_count / total_rows) * 100

        if series.dtype == 'object' or series.dtype == 'string':
            txt = series.dropna().astype(str)
            avg_len = txt.apply(len).mean()
            # Emojis erkennen (alle Nicht-ASCII Zeichen)
            emoji_rows = txt.str.contains(r'[^\x00-\x7F]+', regex=True).sum()
            # Gekürzte Begriffe/Shortforms (z.B. don't, it's, slang 'u', 'w/')
            shortforms = txt.str.contains(r"([a-zA-Z]'[a-zA-Z]|\b[uU]\b|\bw/|\b[bB]4\b)", regex=True).sum()
            url_count = txt.str.contains(r'http[s]?://', regex=True).sum()
            special_chars = txt.apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s,.]', x))).sum()
        else:
            avg_len = emoji_rows = shortforms = url_count = special_chars = "-"

        analysis_data.append({
            'Spalte': col,
            'NaN (Mancos)': nan_count,
            'Unique': unique_count,
            'Kardinalität (%)': round(cardinality, 2),
            'Ø Zeichen': round(avg_len, 1) if isinstance(avg_len, float) else "-",
            'Texte m. Emojis': emoji_rows,
            'Texte m. Kürzeln': shortforms,
            'URLs': url_count,
            'Sonderzeichen': special_chars
        })

    print("\n📊 ÜBERSICHT DER SPALTEN-QUALITÄT:")
    display(pd.DataFrame(analysis_data))
    print('=' * 80)

    # 3. 🧠 STRUKTURELLE TEXT-TIEFE (NEU)
    if txt_col_name in df.columns:
        print("\n🧠 STRUKTURELLE TEXT-ANALYSE (KOMPLEXITÄT & NOISE):")
        txt_series = df[txt_col_name].astype(str)

        # Lexical Diversity & Stopword Ratio
        common_stop = {'the', 'a', 'is', 'in', 'it', 'you', 'i', 'and', 'on', 'for', 'be', 'of'}
        lex_div = txt_series.apply(
            lambda x: len(set(x.lower().split())) / len(x.split()) if len(x.split()) > 0 else 0).mean()
        stop_ratio = txt_series.apply(
            lambda x: sum(1 for w in x.lower().split() if w in common_stop) / len(x.split()) if len(
                x.split()) > 0 else 0).mean()

        depth_df = pd.DataFrame({
            'Metrik': ['Ø Vokabular-Vielfalt (1=hoch)', 'Ø Stoppwort-Last (0-1)', 'Satzzeichen-Spam (!!!)',
                       'Texte mit Zahlen'],
            'Wert': [round(lex_div, 2), round(stop_ratio, 2), txt_series.str.contains(r'[!?]{2,}').sum(),
                     txt_series.str.contains(r'\d').sum()]
        })
        display(depth_df)

        print("\n🔝 TOP 10 ROH-TOKENS (DOMINANTES RAUSCHEN):")
        all_words = " ".join(txt_series).lower().split()
        display(pd.DataFrame(Counter(all_words).most_common(10), columns=['Wort', 'Anzahl']).T)
        print('=' * 80)

    # 4. ⚖️ KLASSEN-VERGLEICH (TARGET)
    if 'target' in df.columns and txt_col_name in df.columns:
        print("\n⚖️ VERGLEICH: MANCOS NACH KLASSE (TARGET):")
        # RAM-schonende Kopie für die Analyse
        df_TEMP = df[[txt_col_name, 'target']].copy()
        df_TEMP['len'] = df_TEMP[txt_col_name].astype(str).apply(len)
        df_TEMP['upper'] = df_TEMP[txt_col_name].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()))

        class_mancos = df_TEMP.groupby('target').agg({
            'len': 'mean',
            'upper': 'mean',
            txt_col_name: lambda x: x.str.contains(r'[^\x00-\x7F]+').sum()
        }).rename(columns={'len': 'Ø Länge', 'upper': 'Ø Großbuchst.', txt_col_name: 'Texte m. Emojis'})

        display(class_mancos.round(2))
        del df_TEMP
        print('=' * 80)

    # 5. 📏 STATISTIK DER ROH-TEXTLÄNGEN (ZEICHEN & WÖRTER)
    if txt_col_name in df.columns:
        print("\n📏 STATISTIK DER ROH-TEXTLÄNGEN:")
        stats_len = df[txt_col_name].astype(str).apply(len).describe().to_frame().T
        stats_words = df[txt_col_name].astype(str).apply(lambda x: len(x.split())).describe().to_frame().T
        stats_len.index, stats_words.index = ['Zeichen-Anzahl'], ['Wort-Anzahl']
        display(pd.concat([stats_len, stats_words]))
        print('=' * 80)

    # 6. ⚠️ ANOMALIEN-BOARD (MANCOS ALS ZAHL)
    print("\n⚠️ GEFUNDENE ANOMALIEN / MANCOS:")
    txt_series = df[txt_col_name].astype(str)
    anomalie_df = pd.DataFrame({
        'Anomalie-Typ': [
            'Leere Texte / Nur Whitespace',
            'Extrem kurz (< 5 Zeichen)',
            'Reine Emoji-Texte (kein Alphabet)',
            'Gekürzte Begriffe (Shortforms)',
            'Shouting (Nur Großbuchstaben)',
            'Text-Duplikate (Spam-Gefahr)',
            'Inkonsistente Umbrüche (\\n)'
        ],
        'Anzahl (Count)': [
            df[txt_series.str.strip() == ""].shape[0],
            df[txt_series.apply(len) < 5].shape[0],
            df[~txt_series.str.contains(r'[a-zA-Z]', na=False)].shape[0],
            txt_series.str.contains(r"([a-zA-Z]'[a-zA-Z]|\b[uU]\b|\bw/)", regex=True).sum(),
            (txt_series.apply(lambda x: x.isupper() if len(x) > 10 else False)).sum(),
            df[txt_col_name].duplicated().sum(),
            txt_series.str.contains(r'\n').sum()
        ],
        'Bedeutung': [
            'Kein Info-Gehalt', 'Kaum Kontext', 'Nicht sprachlich auswertbar',
            'Informelle Sprache', 'Extreme Emotionalität', 'Verzerrt Vokabular', 'Struktur-Rauschen'
        ]
    })
    display(anomalie_df)
    print('=' * 80)


# ==============================================================================
# 2. TEXT PREPROCESSING (KLASSEN & FUNKTIONEN)
# ==============================================================================

class UppercaseWordCount(BaseEstimator, TransformerMixin):
    """
    Feature Engineering: Zählt Wörter, die komplett in Großbuchstaben geschrieben sind.

    Logik:
    Großbuchstaben (CAPS LOCK) sind oft ein Indikator für Dringlichkeit oder Panik
    in Katastrophen-Tweets.
    """

    def __init__(self, min_len=2): self.min_len = min_len

    def fit(self, X, y=None): return self

    def transform(self, X):
        pattern = re.compile(r"\b[A-ZÄÖÜ]{%d,}\b" % self.min_len)
        X_series = pd.Series(X)
        counts = np.array([len(pattern.findall(t)) for t in X_series], dtype=np.float32).reshape(-1, 1)
        return counts


class SparseToDense(BaseEstimator, TransformerMixin):
    """
    Hilfsklasse: Konvertiert Sparse-Matrizen (z.B. von TF-IDF) in Dense-Arrays.

    Warum?
    Manche Modelle (wie GaussianNB oder manche Random Forest Implementierungen)
    können nicht mit Sparse-Matrizen umgehen und benötigen dichte Arrays.
    """

    def __init__(self, dtype=np.float32): self.dtype = dtype

    def fit(self, X, y=None): return self

    def transform(self, X):
        if sparse.issparse(X): return X.toarray().astype(self.dtype, copy=False)
        return np.asarray(X, dtype=self.dtype)


# ==============================================================================
# 1. MODULARE EINZEL-WERKZEUGE zur Reinigung
# ==============================================================================

def module_emoji_demask(text):
    """Wandelt Emojis in Text um (🔥 -> fire)."""
    text = emoji.demojize(str(text), delimiters=(" ", " "))
    return text.replace("_", " ").replace(":", "")


def module_translate(text, ziel_sprache='en'):
    """Erkennt Fremdsprachen und übersetzt sie in die Zielsprache."""
    try:
        if len(text) > 5 and re.search(r'[a-zA-Z]', text):
            lang = detect(text)
            if lang != ziel_sprache:
                return GoogleTranslator(source='auto', target=ziel_sprache).translate(text)
        return text
    except:
        return text


def module_expand_shorts(text):
    """Ersetzt Slang und Kürzel durch Vollwörter (Kontext-Erhalt)."""
    short_map = {
        "can't": "cannot", "won't": "will not", "don't": "do not",
        "it's": "it is", "i'm": "i am", "u ": "you ", " w/": " with",
        "b4": "before", " r ": " are ", " y ": " why ", "%": " percent ", "$": " dollar "
    }
    for short, long in short_map.items():
        text = re.sub(rf"\b{short}\b", long, text, flags=re.IGNORECASE)
    return text


def module_convert_numbers(text, p_engine):
    """Wandelt Zahlen in geschriebene Wörter um (5 -> five)."""
    return re.sub(r'\d+', lambda m: p_engine.number_to_words(m.group(0)), text)


def module_noise_removal(text):
    """Entfernt URLs, Mentions und bereinigt Sonderzeichen."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)  # Behält nur Buchstaben
    return re.sub(r'\s+', ' ', text).strip()


def get_tokens(text, lemmatizer=None, stop_words=None):
    """
    DER BASIS-TOKENIZER (Modularer Kern).
    Wandelt Text in eine Liste um (für Word2Vec oder Lemmatisierung).
    """
    text = module_noise_removal(str(text))
    tokens = word_tokenize(text)

    # Optional: Lemmatisierung (falls Instanz übergeben)
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(w) for w in tokens if len(w) > 1 or w in ['i', 'a']]

    # Optional: Stoppwort-Filterung
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]

    return tokens


def lemmatize_text(text, lemmatizer=None):
    """
    DER TEXT-BAUER (Nutzt get_tokens).
    Gibt einen sauberen String zurück (für das LSTM-Modell).
    """
    # Ruft get_tokens auf -> Höchste Wiederverwendbarkeit
    tokens = get_tokens(text, lemmatizer=lemmatizer)
    return " ".join(tokens)


# ==============================================================================
# 2. DIE HAUPT-PIPELINE (Reinigung Manager-Funktion)
# ==============================================================================

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn kompatibler Transformer zur Textbereinigung.
    Nutzt die modularen Funktionen.
    """

    def __init__(self, stop_words=None):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stop_words
        self.p_engine = inflect.engine()

    def fit(self, X, y=None): return self

    def transform(self, X):
        # Nutzt Parallelisierung für Performance
        def process_single(text):
            t = module_emoji_demask(text)
            t = module_expand_shorts(t)
            t = module_convert_numbers(t, self.p_engine)
            t = lemmatize_text(t, self.lemmatizer)
            return t

        return [process_single(x) for x in X]


def standard_Cleare_TEXT(df: pd.DataFrame, ziel_sprache: str = 'en'):
    """
    NLP-REINIGUNGS-PIPELINE:
    - Bereinigt Mancos (NaNs, Duplikate, Slang, Emojis).
    """
    # Initialisierung
    DetectorFactory.seed = 42
    p = inflect.engine()
    lemmatizer = WordNetLemmatizer()
    txt_col = 'text'
    new_name = "df_Cleaning"
    print(f'🚀 START NLP-PIPELINE: {new_name.upper()}')
    print(f"DEBUG: Spalten vor Reinigung: {list(df.columns)}")
    print('=' * 80)
    df_TEMP = df.copy()

    # 2. Parallelverarbeitung
    def process_text(text):
        t = module_translate(module_emoji_demask(text), ziel_sprache)
        t = module_convert_numbers(module_expand_shorts(t), p)
        t = lemmatize_text(t, lemmatizer)
        return t

    print(f"🧠 Verarbeite {len(df_TEMP)} Zeilen parallel (Core-Reserve aktiv)...")
    with tqdm_joblib(tqdm(total=len(df_TEMP), desc="NLP Pipeline", **TQDM_STYLE)) as pbar:
        results = Parallel(n_jobs=-2)(
            delayed(process_text)(txt) for txt in df_TEMP[txt_col]
        )
    df_TEMP['cleaned_text'] = results

    # 3. Quality Check (Mancos entfernen)
    df_TEMP = df_TEMP.drop_duplicates(subset=['cleaned_text'])
    df_TEMP = df_TEMP[df_TEMP['cleaned_text'].str.len() > 2].reset_index(drop=True)

    globals()[new_name] = df_TEMP.copy()

    # Statistische Tabelle zur Unterstützung der Visualisierung
    stats_data = {
        "Metrik": ["Zeilen Vorher", "Zeilen Nachher", "Duplikate entfernt"],
        "Wert": [len(df), len(df_TEMP), len(df) - len(df_TEMP)]
    }
    print("\n📊 ZUSAMMENFASSUNG DER REINIGUNG:")
    print(pd.DataFrame(stats_data).to_string(index=False))
    del df_TEMP
    import gc
    gc.collect()
    print(f"✅ REINIGUNG BEENDET: {new_name} ist bereit für Radar-Plot Signale.")
    return globals()[new_name]


# ==============================================================================
# 3. FEATURE ENGINEERING (LOGIK & METRIKEN)
# ==============================================================================

def get_strategic_stopwords(custom_Stop_word=None, custom_Stop_word_remane=None):
    """
    ENGINEERING FUNKTION:
    - custom_Stop_word: Liste von Wörtern, die zusätzlich als RAUSCHEN entfernt werden.
    - custom_Stop_word_remane: Liste von Wörtern, die als SIGNAL erhalten bleiben.
    """
    nltk.download('stopwords', quiet=True)

    # 1. Basis-Rauschen (NLTK + Deine Ergänzungen)
    standard_noise = {'a', 'i', 'you', 'my', 'be', 'with', 'have', 'like', 'it', 'this',
                      'the', 'of', 'on', 'and', 'that', 'for', '-', 'is', 'in', 'to'}

    stop_words_combined = set(stopwords.words('english')).union(standard_noise)

    if custom_Stop_word:
        stop_words_combined.update(custom_Stop_word)

    # 2. Strategische Signale (Werden aus der Löschliste gerettet)
    starke_indikatoren = ['after', 'are', 'as', 'at', 'by', 'from', 'between', 'during', 'into', 'while']

    if custom_Stop_word_remane:
        starke_indikatoren.extend(custom_Stop_word_remane)

    # 3. Engineering: Signale aus der Liste entfernen (discard)
    for wort in starke_indikatoren:
        stop_words_combined.discard(wort)

    # Erfolgskontrolle (Tabelle)
    stats = pd.DataFrame({"Kategorie": ["Aktive Filter (Noise)", "Gerettete Signale (Features)"],
                          "Anzahl": [len(stop_words_combined), len(starke_indikatoren)]})

    print("\n🛡️ SIGNAL-ENGINEERING KONFIGURATION:")
    print(stats.to_markdown(index=False))

    print(f"\n💡 Beispiele geretteter Signale: {starke_indikatoren[:5]}...")
    return stop_words_combined
    del stats


def run_LLM_enclave_call_offline(df):
    """
    1. Nutzt globale ENCLAVE_DIR Pfade.
    2. Überwacht die gesamte Struktur (Reihen, Spalten, Größe) + Master-Spalte (Text-Hash).
    3. Erstellt Schablone oder lädt sie blitzschnell.
    """
    # 1. KONFIGURATION & PFADE
    CACHE_FILE = os.path.join(ENCLAVE_DIR, "enclave_schablone.parquet")
    METADATA_FILE = os.path.join(ENCLAVE_DIR, "cache_fingerprint.txt")

    # 2. FINGERABDRUCK (Prüft ob Daten zum Cache passen)
    current_rows = len(df)
    text_data = "".join(df[COLS['text']].astype(str))
    text_hash = hashlib.md5(text_data.encode('utf-8')).hexdigest()
    current_fingerprint = f"R:{current_rows}|H:{text_hash}"

    # 3. CACHE-CHECK & LADEN
    loaded_from_cache = False
    if os.path.exists(CACHE_FILE) and os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            saved_fingerprint = f.read().strip()

        if saved_fingerprint == current_fingerprint:
            print(f"📦 Integrität bestätigt ({current_rows} Zeilen). Lade Schablone...")
            df_schablone = pd.read_parquet(CACHE_FILE)
            if 'enclave_score' in df.columns:
                df = df.drop(columns=['enclave_score'])

            # SICHERER JOIN: Verknüpft über den Index
            df = df.merge(df_schablone[['enclave_score']], left_index=True, right_index=True, how='left')
            loaded_from_cache = True
        else:
            print("🚨 System-Änderung erkannt (z.B. Sampling). Schablone veraltet.")

    # 4. NEUBERECHNUNG
    if not loaded_from_cache:
        selected_model = setup_enclave()
        if not selected_model:
            print("🛑 Abbruch: Enclave-Umgebung nicht bereit.")
            df['enclave_score'] = -1
        else:
            print(f"🚀 Starte LLM-Engineering (Ollama: {selected_model}) für {current_rows} Tweets...")
            tqdm.pandas(desc=f"Enclave ({selected_model})", **TQDM_STYLE)
            df['enclave_score'] = df[COLS['text']].progress_apply(
                lambda x: LLM_enclave_call_offline(x, selected_model)
            )
            if not os.path.exists(ENCLAVE_DIR): os.makedirs(ENCLAVE_DIR)
            df[['enclave_score']].to_parquet(CACHE_FILE)
            with open(METADATA_FILE, "w") as f:
                f.write(current_fingerprint)
            print(f"💾 Schablone sicher in {ENCLAVE_DIR} abgelegt.")

    # 5. RÜCKGABE & ÜBERSICHT
    print("\n📊 ENCLAVE STATUS-BERICHT (Numerische Unterstützung):")
    display(df['enclave_score'].value_counts().to_frame())
    return df


def LLM_0_1_bool(text):
    """
    Simuliert einen LLM-Score (Large Language Model).
    Erzeugt ein starkes Feature basierend auf semantischem Verständnis.
    Logik:
    Prüft auf extrem starke Signalwörter (STRONG_LLM_SIGNALS).
    Gibt 0.8 für sehr wahrscheinlich, 0.5 für unsicher zurück.
    (In Produktion würde hier ein API-Call zu GPT/Ollama stehen).
    """
    text_lower = str(text).lower()
    if any(s in text_lower for s in STRONG_LLM_SIGNALS): return 0.8
    return 0.5


def Nicht_Katastrophe(text):
    """
    Prüft den Text gegen die umfangreiche NOISE_KEYWORDS Liste (eBay, Gaming, Metaphern).
    Gibt 1 zurück, wenn es sich wahrscheinlich um Rauschen handelt.
    """
    text_lower = str(text).lower()
    return any(word in text_lower for word in NOISE_KEYWORDS)


def run_signal_noise_study(df, text_col='cleaned_text', target_col='target'):
    """
    STUDIE TEIL 1: SIGNAL- UND RAUSCH-ANALYSE
    Identifiziert Wörter mit der höchsten Relevanz für Katastrophen.
    """
    print("STUDIE TEIL 1: SIGNAL- UND RAUSCH-ANALYSE")
    # RAM-Schonende Kopie für die Analyse
    df_TEMP = df[[text_col, target_col]].copy()  # [Regel 2025-11-21]
    words_disaster = " ".join(df_TEMP.loc[df_TEMP[target_col] == 1, text_col]).split()
    words_normal = " ".join(df_TEMP.loc[df_TEMP[target_col] == 0, text_col]).split()
    freq_disaster = Counter(words_disaster)
    freq_normal = Counter(words_normal)

    relevance_data = []
    for word in freq_disaster:
        f_dis = freq_disaster[word]
        f_norm = freq_normal[word]
        total = f_dis + f_norm
        if total > 5:
            relevance_data.append({
                'Wort': word,
                'Relevanz': round(f_dis / (f_norm + 1), 2),
                'Total': total,
                'Disaster_Freq': f_dis
            })
    df_study = pd.DataFrame(relevance_data).sort_values(by='Relevanz', ascending=False)

    print("\n📊 TOP 10 SIGNALWÖRTER (KATASTROPHE):")
    display(df_study.head(10))  # Numerische Unterstützung
    del df_TEMP, relevance_data
    gc.collect()
    return df_study


def calculate_exponential_disaster_score(text):
    """
    1. Prüft auf DISASTER_KEYWORDS.
    2. Wenn Noise erkannt wird (Nicht_Katastrophe), wird das Gewicht reduziert.
    3. Belohnt N-Gramme (aufeinanderfolgende Keywords) exponentiell (2^x).
    """
    is_noise = Nicht_Katastrophe(text)
    words = str(text).lower().split()
    score = 0
    for i in range(len(words)):
        if words[i] in DISASTER_KEYWORDS:
            current_weight = 4
            if is_noise: current_weight = 1
            score += current_weight
            # Bonus für Wortketten
            if i + 1 < len(words) and words[i + 1] in DISASTER_KEYWORDS:
                score += 8 if not is_noise else 2
                if i + 2 < len(words) and words[i + 2] in DISASTER_KEYWORDS:
                    score += 16 if not is_noise else 4
    return float(score)


def process_emojis_and_context(text):
    """
    Ablauf:
    1. Emojis -> Text (mit EMOJI_REPLACEMENTS Mapping für semantische Bedeutung).
    2. Extraktion von Trigrammen (3 Wörter) rund um Katastrophen-Keywords.
       Dies hilft dem Modell, den Kontext von "Fire" (Konzert vs. Waldbrand) zu verstehen.
    """
    text = emoji.demojize(str(text), delimiters=(" ", " "))
    for emo, word in EMOJI_REPLACEMENTS.items(): text = text.replace(emo, word)
    words = text.lower().split()
    trigrams = []
    for i, w in enumerate(words):
        if any(k in w for k in DISASTER_KEYWORDS):
            chunk = words[max(0, i - 1):min(len(words), i + 2)]
            if len(chunk) >= 1: trigrams.append(" ".join(chunk))
    return text, " ".join(trigrams)


def get_uppercase_token_count(text, min_len=2):
    """Zählt Tokens, die komplett großgeschrieben sind (für EDA)."""
    tokens = str(text).split()
    pattern = re.compile(r"^[A-ZÄÖÜ]{%d,}$" % min_len)
    return sum(1 for t in tokens if pattern.match(t))


def add_tweet_metrics(df, column='text', column_clean='cleaned_text'):
    """
    Kombiniertes Feature Engineering:
    - length: Zeichenanzahl (Wichtig für Ensemble-KeyError Fix)
    - word_count: Wortanzahl
    - caps_count: Großbuchstaben (Dringlichkeits-Signal)
    - subjectivity: Semantische Einordnung via TextBlob
    """
    df_TEMP = df.copy()
    print(f"🚀 Berechne Metriken (Raw-Quelle: {column}, Clean-Quelle: {column_clean})...")

    # 1. BASIS-ZÄHLER auf dem ORIGINAL-TEXT (column_raw)
    raw_text_series = df_TEMP[column].astype(str).fillna('')
    df_TEMP['length'] = raw_text_series.str.len().astype(int)
    df_TEMP['caps_count'] = raw_text_series.apply(lambda x: sum(1 for c in x if c.isupper())).astype(int)
    df_TEMP['word_count'] = raw_text_series.apply(lambda x: len(x.split())).astype(int)

    # 2. SEMANTIK auf dem CLEANED-TEXT
    if column_clean in df_TEMP.columns:
        df_TEMP['subjectivity'] = df_TEMP[column_clean].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity
        ).astype(float)
    else:
        df_TEMP['subjectivity'] = raw_text_series.apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity
        ).astype(float)

    gc.collect()
    return df_TEMP


def clip_counts(s, max_val=5):
    """Hilfsfunktion für Histogramme: Begrenzt Werte auf max_val (Outlier-Handling)."""
    return s.clip(upper=max_val)


def get_punctuation_metrics(text):
    """
    Misst die Dringlichkeit basierend auf Satzzeichen.
    Gibt die Anzahl von ! und ? sowie die Punkt-Dichte zurück.
    """
    text = str(text)
    excl = text.count('!')
    ques = text.count('?')
    # Punkte deuten oft auf sachliche News-Ticker hin
    dots = text.count('.')
    return pd.Series([excl, ques, dots])


def get_source_credibility(text):
    """
    Prüft auf Merkmale professioneller Berichterstattung.
    1 = Enthält Link oder typische News-Begriffe (Breaking, News, Report).
    """
    text_lower = str(text).lower()

    # 1. URL Check (bleibt gleich)
    has_url = 1 if re.search(r'https?://\S+|www\.\S+', text) else 0

    # Check auf Begriffe
    combined_keywords = TIME_KEYWORDS + NEWS_KEYWORDS
    term_match = any(word in text_lower for word in combined_keywords)

    # Check auf UHRZEIT
    time_match = bool(re.search(r'\d{1,2}:\d{2}|\d{1,2}\s?(am|pm)', text_lower))

    # Wenn ein Begriff ODER eine Uhrzeit gefunden wurde -> is_news_style = 1
    is_news_style = 1 if (term_match or time_match) else 0

    return pd.Series([has_url, is_news_style])


def get_hit_keyword(text):
    t = str(text).lower()
    for kw in DISASTER_KEYWORDS:
        if kw in t: return kw
    return None


def signal_type(text):
    """
    1.0= Disaster Match
    -1.0= Noise Match (Spam/Gaming)
    0.0= Kein Treffer (Neutral)
    """
    t = str(text).lower()

    # 1. Check auf echte Katastrophen (Priorität)
    if any(kw in t for kw in DISASTER_KEYWORDS):
        return 1.0

    # 2. Check auf Noise/Spam
    if any(kw in t for kw in NOISE_KEYWORDS):
        return -1.0

    # 3. Weder noch
    return 0.0


def get_hit_location(text):
    all_locs = SITUATIVE_LOCATIONS + POI_LOCATIONS
    t = str(text).lower()
    for loc in all_locs:
        if loc in t: return loc
    return None


def signal_location(text, loc_profile):
    """
    ERWEITERTES LOCATION-SCORING:
    - Nutzt situative, relative und POI-Begriffe.
    - Erkennt Augenzeugen durch Deiktika (hier, gegenüber).
    """
    t = str(text).lower()
    lp = str(loc_profile).lower()

    # 1. Höchste Priorität: Profil-Verifikation (1.5)
    if lp != 'nan' and lp != '' and len(lp) > 2 and lp in t:
        return 1.5
    # 2. Zweite Priorität: Kritische Infrastruktur / POI (1.0)
    if any(p in t for p in POI_LOCATIONS):
        return 1.0
    # 3. Dritte Priorität: Situative Orte (Kitchen, Street) (0.9)
    if any(s in t for s in SITUATIVE_LOCATIONS):
        return 0.9
    # 4. Vierte Priorität: Relative Begriffe / Augenzeuge (0.8)
    if any(r in t for r in RELATIVE_LOCATIONS):
        return 0.8
    # 5. Fallback: Nichts gefunden
    return 0.0


def signal_time(text):
    """Prüft auf Zeit (Wann). Nutzt globale TIME_KEYWORDS."""
    t = str(text).lower()
    return 1.0 if any(tm in t for tm in TIME_KEYWORDS) else 0.0


def extract_disaster_triad(row):
    """
    Zentrale Logik-Zusammenführung:
    Erzeugt Scores für Typ, Ort, Zeit und ein kombiniertes Triple-Signal.
    """
    txt = row.get('text', '')
    loc = row.get('location', '')

    # Berechnung der Einzel-Signale
    is_type = signal_type(txt)
    is_loc = signal_location(txt, loc)
    is_time = signal_time(txt)

    # Logische Verknüpfung: Alle 3 müssen zutreffen (Intersection)
    is_triple = 1.0 if (is_type + is_loc + is_time == 3.0) else 0.0

    return pd.Series([is_type, is_loc, is_time, is_triple])


# ==============================================================================
# 3.1.1 Advanced Engineering Pipeline zusammen Führung aller neuen Spalte
# ==============================================================================

def apply_advanced_engineering(df):
    """
    ULTIMATIVES ENGINEERING
    - Integriert Triaden-Logik (Type-Loc-Time) & Enclave-Schablonen-System.
    """
    df_TEMP = df.copy()
    tqdm.pandas(desc="Gesamtfortschritt Engineering", **TQDM_STYLE)

    print(f"\n🚀 STARTE ADVANCED FEATURE ENGINEERING (OS: {platform.system()})")

    # SCHRITT 0: BASIS METRIKEN & SEMANTIK
    # Berechnet: length, word_count, caps_count, subjectivity
    df_TEMP = add_tweet_metrics(df_TEMP, column=COLS['text'], column_clean='cleaned_text')

    # 1. KONTEXT & EMOJIS & NOISE
    print("🔍 Schritt 1: Kontext-Extraktion...")
    context_data = df_TEMP[COLS['text']].apply(process_emojis_and_context)
    df_TEMP['clean_text_context'] = context_data.apply(lambda x: x[0])
    df_TEMP['emoji_count'] = context_data.apply(lambda x: len(x[1]) if x[1] else 0)

    # 2. SATZZEICHEN & DRINGLICHKEIT
    print("📝 Schritt 2: Punctuation & Caps Metrics...")
    df_TEMP[['excl_count', 'ques_count', 'dot_count']] = df_TEMP[COLS['text']].apply(get_punctuation_metrics)

    # 3. SOURCE CREDIBILITY (URLs)
    print("🔗 Schritt 5: Source Credibility...")
    df_TEMP[['has_url', 'is_news_style']] = df_TEMP[COLS['text']].apply(get_source_credibility)

    # 4. DIE KATASTROPHEN-TRIADE
    print("🎯 Schritt 6: Erzeuge Triaden-Signale (Type, Location, Time)...")
    triad_cols = ['type_K', 'type_Location', 'type_time', 'type_K_L_t_combi']
    df_TEMP[triad_cols] = df_TEMP.apply(extract_disaster_triad, axis=1)

    # Hilfs-Flag für das finale Scoring
    df_TEMP['is_triple_signal'] = (df_TEMP['type_K_L_t_combi'] > 2).astype(int)

    # 5. NOISE-FILTER (eBay, Gaming, etc.)
    print("📈 Schritt 7: Berechne Noise-Flag... ")
    df_TEMP['is_noise_flag'] = df_TEMP[COLS['text']].apply(Nicht_Katastrophe).astype(int)

    # 6. FINALES SCORING (Inklusive Triple-Signal Gewichtung)
    print("📈 Schritt 8: Berechne finalen Disaster-Score...")
    df_TEMP['disaster_score'] = df_TEMP.apply(
        lambda r: calculate_exponential_disaster_score(r[COLS['text']]) + (r['is_triple_signal'] * 2), axis=1)

    # 7. ENCLAVE KI (Hardware-optimiert für MacBook Air)
    print("🤖 Schritt 9: Enclave LLM Analyse (Core-Reserve aktiv)...")
    # prüft die Schablone, lädt sie oder startet Ollama nur wenn nötig.
    df_TEMP = run_LLM_enclave_call_offline(df_TEMP)

    # Numerische Tabelle zur Unterstützung der Visualisierung
    print("\n📊 ENGINEERING ABGESCHLOSSEN. Numerische Übersicht für Radar-Plot:")
    plot_metrics = ['type_K', 'type_Location', 'type_time', 'type_K_L_t_combi', 'is_triple_signal', 'disaster_score',
                    'is_noise_flag']  # ['enclave_score']
    print(df_TEMP[plot_metrics].describe().loc[['mean', 'min', 'max']])
    global df_Cleaning
    df_Cleaning = df_TEMP.copy()

    del df_TEMP
    gc.collect()
    return df_Cleaning


# ==============================================================================
# 3.1.2 Advanced Engineering Pipeline
# ==============================================================================
def run_full_disaster_pipeline(df, show_plots=True):
    """
    Zentrale Steuerung:
    Erzwingt die sequentielle Abarbeitung aller Engineering-Schritte.
    """
    # SCHRITT 1: DAS KOMPLETTE ENGINEERING STARTEN
    # (Hier kommen nacheinander: Punctuation, Triaden, Kontext und Schritt 7: Enclave)
    df_TEMP = apply_advanced_engineering(df.copy())

    # SCHRITT 2: DER POLAR PLOT (Erzwungen nach dem Engineering)
    if show_plots:
        print("\n📊 Generiere Polar-Plot basierend auf den neuen Features...")
        plot_automated_polar_engineering(df_TEMP)

    # SCHRITT 3: FINALE ÜBERGABE
    global df_Cleaning
    if 'df_Cleaning' in globals():
        del df_Cleaning
        gc.collect()
    df_Cleaning = df_TEMP.copy()
    print("\n📋 Vorschau der ersten 5 Zeilen (df_Cleaning):")
    del df_TEMP
    return df_Cleaning


# ==============================================================================
# 4. VEKTORDATENBANK & EMBEDDINGS (MILVUS & CO.)
# ==============================================================================

class MilvusWrapper:
    """
    Wrapper für die Milvus Vektordatenbank.
    Ermöglicht das Speichern und Suchen von Text-Embeddings.
    """

    def __init__(self, db_name="milvus_demo.db", collection_name="tweets"):
        if MilvusClient is None:
            print("Warnung: pymilvus nicht installiert. Wrapper inaktiv.")
            self.client = None
            return
        self.client = MilvusClient(db_name)
        self.collection_name = collection_name
        self.embedding_fn = milvus_model.DefaultEmbeddingFunction()

    def create_collection(self, dim=768):
        if not self.client: return
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self.client.create_collection(collection_name=self.collection_name, dimension=dim)

    def insert_texts(self, texts, metadata_list=None):
        if not self.client: return
        vectors = self.embedding_fn.encode_documents(texts)
        data = []
        for i, text in enumerate(texts):
            entry = {"id": i, "vector": vectors[i], "text": text}
            if metadata_list:
                entry.update(metadata_list[i])
            data.append(entry)
        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query, limit=3):
        if not self.client: return []
        query_vectors = self.embedding_fn.encode_queries([query])
        return self.client.search(collection_name=self.collection_name, data=query_vectors, limit=limit,
                                  output_fields=["text"])


def calculate_similarity_metrics(embedding_model, data, label_column='target'):
    """
    Berechnet Ähnlichkeiten zwischen Textpaaren (für Duplikat-Erkennung).
    Erwartet DataFrame mit 'text1' und 'text2' Spalten.
    """
    similarities = []
    for _, row in data.iterrows():
        # Dummy-Implementierung für TF-IDF Vektoren (da wir keine Embeddings haben)
        # In einer echten App würde man hier Embeddings nutzen
        v1 = embedding_model.transform([row['text1']])
        v2 = embedding_model.transform([row['text2']])
        sim = (v1 * v2.T).toarray()[0][0]
        similarities.append(sim)
    return similarities


# ==============================================================================
# 5. VISUALISIERUNG (PLOTLY) EDA
# ==============================================================================
def plot_automated_polar_engineering(df, width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height'],
                                     color1=PLOT_COLORS['disaster'], color2=PLOT_COLORS['no_disaster']):
    """
    Vollständiger Polar-Plot mit intelligenter 0-10 Skalierung.
    Behält alle ursprünglichen Sektionen bei, korrigiert aber die visuelle Verzerrung.
    """
    # 1. RAM-schonende Kopie
    df_TEMP = df.copy()

    # 2. Automatische Feature-Erkennung
    exclude = ['target', 'id', 'index', 'Unnamed: 0']
    feature_cols = [c for c in df_TEMP.select_dtypes(include=[np.number]).columns if c not in exclude]

    # 3. DIE BERECHNUNG DER WICHTIGKEIT & SKALIERUNG (0-10 Engine)
    stats = df_TEMP.groupby('target')[feature_cols].mean()

    # Neu: Wir berechnen die Skalierung basierend auf den Spalten-Informationen (Min/Max)
    scaled_stats_list = []
    importance_map = {}

    for col in feature_cols:
        g_min = df_TEMP[col].min()
        g_max = df_TEMP[col].max()
        g_range = g_max - g_min

        m0, m1 = stats.loc[0, col], stats.loc[1, col]

        # Normierung auf 0-10 für JEDE Spalte (egal ob -1/1 oder 0-500)
        if g_range > 0:
            s0 = ((m0 - g_min) / g_range) * 10
            s1 = ((m1 - g_min) / g_range) * 10
        else:
            s0, s1 = 0, 0

        scaled_stats_list.append({'Feature': col, 'target': 0, 'Score': s0})
        scaled_stats_list.append({'Feature': col, 'target': 1, 'Score': s1})
        # Wichtigkeit für die Sortierung (Delta auf 10er Skala)
        importance_map[col] = abs(s1 - s0)

    df_scaled = pd.DataFrame(scaled_stats_list)
    df_scaled['Kategorie'] = df_scaled['target'].map({1: 'Katastrophe', 0: 'Keine Katastrophe'})

    # 4. SORTIER-LOGIK (Fächer-Effekt nach Trennschärfe)
    # Erstellt die Reihenfolge basierend auf der Signalstärke
    sorted_features = pd.Series(importance_map).sort_values(ascending=False).index.tolist()

    # WICHTIG: Den DataFrame sortieren, damit die Linien nicht kreuz und quer springen
    df_scaled['Feature'] = pd.Categorical(df_scaled['Feature'], categories=sorted_features, ordered=True)
    df_scaled = df_scaled.sort_values(['target', 'Feature'])

    # 5. POLAR PLOT (0-10 Skala)
    fig_polar = px.line_polar(
        df_scaled, r='Score', theta='Feature', color='Kategorie', line_close=True,
        template=PLOT_CONFIG['template'],
        color_discrete_map={'Katastrophe': color1, 'Keine Katastrophe': color2},
        category_orders={"Feature": sorted_features},  # Erzeugt den Fächer
        title="<b>Signal-Profil (Skala 0-10)</b>"
    )

    fig_polar.update_traces(fill='toself', opacity=0.6)
    fig_polar.update_layout(
        width=width, height=height, title_x=0.5,
        paper_bgcolor=PLOT_COLORS['background'],
        plot_bgcolor=PLOT_COLORS['background'],
        polar=dict(
            bgcolor=PLOT_COLORS['background'],
            radialaxis=dict(visible=True, range=[0, 10], gridcolor=PLOT_COLORS['grid'], ticksuffix="pt"),
            angularaxis=dict(direction="clockwise", gridcolor=PLOT_COLORS['grid'])
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, title=None)
    )
    fig_polar.show()

    # 6. FEATURE IMPORTANCE BARPLOT (Originale Delta-Logik beibehalten)
    importance_df = pd.DataFrame({
        'Feature': list(importance_map.keys()),
        'Signal_Stärke': list(importance_map.values())
    }).sort_values(by='Signal_Stärke', ascending=False)

    fig_bar = px.bar(
        importance_df, x='Signal_Stärke', y='Feature', orientation='h',
        title="Ranking der Trennschärfe (Delta auf 10er Basis)",
        template=PLOT_CONFIG['template'],
        color='Signal_Stärke',
        color_continuous_scale=PLOT_COLORS['scale_two']
    )

    fig_bar.update_layout(
        width=width, height=height * 0.8,
        paper_bgcolor=PLOT_COLORS['background'],
        plot_bgcolor=PLOT_COLORS['background'],
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False
    )
    fig_bar.show()

    # 7. NUMERISCHE TABELLE (Unterstützungs-Regel)
    print("\n📊 SORTIERTE BASISDATEN (Mittelwerte & 0-10 Analyse):")
    stats_out = stats.T
    stats_out.columns = ['Ø Keine Katastrophe', 'Ø Katastrophe']
    stats_out['Delta_Score_0_10'] = stats_out.index.map(importance_map)
    display(stats_out.loc[sorted_features].round(4))

    # 8. FINALER RAM-CLEANUP
    del df_TEMP, df_scaled, importance_df, stats_out
    gc.collect()

    return "EDA abgeschlossen"


def plot_feature_signal_ratio(df, feature_col, top_n=30,
                              color1=PLOT_COLORS['disaster'], color2=PLOT_COLORS['no_disaster'],
                              width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Erstellt eine Signal-Analyse-Bar-Chart
    """
    df_TEMP = df.copy()

    if feature_col not in df_TEMP.columns:
        print(f"⚠️ Warnung: Spalte '{feature_col}' nicht gefunden.")
        return

    # Datenbereinigung & Statistik
    df_TEMP[feature_col] = df_TEMP[feature_col].fillna('Unknown')
    if df_TEMP[feature_col].dtype == 'object':
        df_TEMP[feature_col] = df_TEMP[feature_col].astype(str).str.replace('%20', ' ')

    stats = df_TEMP.groupby(feature_col)[COLS['target']].agg(['mean', 'count'])
    top_stats = stats[stats['count'] > 5].sort_values(by='mean', ascending=False).head(top_n).reset_index()

    if top_stats.empty:
        print(f"⚠️ Keine Daten für '{feature_col}' mit Count > 5 gefunden.")
        return

    top_stats['Signal'] = top_stats['mean'].apply(lambda x: 'Gefahr' if x > 0.5 else 'Sicher')

    # Plot Erstellung
    fig = px.bar(top_stats, x='mean', y=feature_col, orientation='h', color='Signal',
                 title=f"Signal-Analyse: {feature_col}",
                 color_discrete_map={"Gefahr": color1, "Sicher": color2}, template=PLOT_CONFIG['template'])

    # Automatische prozentuale Anpassung der Margins
    m_left = int(width * 0.15)
    m_right = int(width * 0.05)
    m_top = int(height * 0.12)
    m_bottom = int(height * 0.08)

    # Layout-Anpassung
    fig.update_layout(width=width, height=height, paper_bgcolor=PLOT_COLORS['background'],
                      plot_bgcolor=PLOT_COLORS['background'],
                      yaxis={'categoryorder': 'total ascending'}, margin=dict(l=m_left, r=m_right, t=m_top, b=m_bottom),
                      title_x=0.5)
    fig.show()

    # Numerische Daten als DataFrame anzeigen (Vorgabe: Tabelle zur Visualisierung)
    print(f"\n📊 DATENTABELLE: {feature_col}")
    display(top_stats.drop(columns=['Signal']).head(10))
    del df_TEMP


def plot_metric_by_target(df_input, metric_col, color1=PLOT_COLORS['disaster'], color2=PLOT_COLORS['no_disaster'],
                          width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Erstellt ein Histogramm mit Violin-Plot, skaliert Layout-Elemente prozentual
    und nutzt die globale PLOT_CONFIG.
    """
    # RAM-Schutz: Kopie erstellen
    df_TEMP = df_input.copy()

    # Sicherstellen, dass die Metrik-Spalte existiert
    if metric_col not in df_TEMP.columns:
        if metric_col == 'length':
            df_TEMP['length'] = df_TEMP[COLS['text']].apply(len)
        else:
            print(f"⚠️ Warnung: Spalte '{metric_col}' nicht gefunden.")
            return

    # Explizite Benennung für die Farbzuteilung
    df_TEMP['Status'] = df_TEMP[COLS['target']].map({1: 'Katastrophe', 0: 'Keine Katastrophe'})

    fig = px.histogram(df_TEMP, x=metric_col, color='Status', marginal="violin", barmode='overlay', opacity=0.65,
                       color_discrete_map={"Katastrophe": color1, "Keine Katastrophe": color2},
                       template=PLOT_CONFIG['template'], title=f"Verteilung: {metric_col} nach Zielklasse")

    # Automatische prozentuale Anpassung der Margins
    m_left = int(width * 0.08)
    m_right = int(width * 0.05)
    m_top = int(height * 0.12)
    m_bottom = int(height * 0.10)

    fig.update_layout(width=width, height=height, paper_bgcolor=PLOT_COLORS['background'],
                      plot_bgcolor=PLOT_COLORS['background'],
                      margin=dict(l=m_left, r=m_right, t=m_top, b=m_bottom), title_x=0.5,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.show()

    print(f"\n📊 STATISTIK: {metric_col}")
    stats_df = df_TEMP.groupby('Status')[metric_col].describe().reset_index()
    display(stats_df)

    del df_TEMP


def _update_layout(fig, title=None, xaxis=None, yaxis=None,
                   width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Dynamische Legenden-Positionierung basierend auf Höhe.
    """
    # 1. Dynamische Margins (Vorgabe: Keine festen Pixelwerte)
    m_left = int(width * 0.12)  # Etwas mehr Platz für Y-Achsen-Beschriftung
    m_right = int(width * 0.05)
    m_top = int(height * 0.15)
    m_bottom = int(height * 0.12)

    # 2. Haupt-Layout Update
    fig.update_layout(template=PLOT_CONFIG['template'], paper_bgcolor=PLOT_COLORS['background'],
                      plot_bgcolor=PLOT_COLORS['background'], font_color=PLOT_COLORS['text'],
                      title={'text': f"<b>{title}</b>" if title else "", 'y': 0.95, 'x': 0.5, 'xanchor': 'center',
                             'yanchor': 'top'},
                      width=width, height=height,
                      legend=dict(font=dict(color=PLOT_COLORS['text'], size=min(11, int(width * 0.025))),
                                  orientation="h", yanchor="bottom",
                                  y=-0.22 if height < 450 else -0.18, xanchor="center", x=0.5),
                      margin=dict(l=m_left, r=m_right, t=m_top, b=m_bottom))
    if xaxis:
        if isinstance(xaxis, dict):
            fig.update_xaxes(**xaxis)
        else:
            fig.update_xaxes(title_text=str(xaxis))

    if yaxis:
        if isinstance(yaxis, dict):
            fig.update_yaxes(**yaxis)
        else:
            fig.update_yaxes(title_text=str(yaxis))
    fig.update_xaxes(gridcolor=PLOT_COLORS['grid'], zerolinecolor=PLOT_COLORS['grid'], showline=True, linewidth=1,
                     linecolor=PLOT_COLORS['grid'])
    fig.update_yaxes(gridcolor=PLOT_COLORS['grid'], zerolinecolor=PLOT_COLORS['grid'], showline=True, linewidth=1,
                     linecolor=PLOT_COLORS['grid'])
    return fig


def plot_target_distribution(df, save_path=None,
                             color1=PLOT_COLORS['disaster'], color2=PLOT_COLORS['no_disaster'],
                             width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Plottet die Verteilung der Zielvariable (0 vs 1).
    """
    counts = df[COLS['target']].value_counts().reset_index()
    counts.columns = ['target', 'count']
    counts['label'] = counts['target'].map({0: 'Keine Katastrophe', 1: 'Katastrophe'})
    total = counts['count'].sum()
    counts['Anteil (%)'] = ((counts['count'] / total) * 100).round(2)

    fig = px.bar(counts, x='label', y='count', color='label',
                 color_discrete_map={'Keine Katastrophe': color2, 'Katastrophe': color1},
                 text='count')

    _update_layout(fig, title='Verteilung der Zielvariable', width=width, height=height, xaxis={'title': 'Klasse'},
                   yaxis={'title': 'Anzahl Tweets'})

    # Text auf den Balken formatieren
    fig.update_traces(textposition='outside')

    if save_path: fig.write_image(save_path)
    fig.show()

    print("\n📊 NUMERISCHE VERTEILUNG:")
    display(counts[['label', 'count', 'Anteil (%)']])


def plot_tweet_length_distribution(df, save_path=None,
                                   color1=PLOT_COLORS['disaster'], color2=PLOT_COLORS['no_disaster'],
                                   width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Plottet ein Histogramm der Tweet-Längen
    """
    # Sicherstellen, dass die Längen-Spalte existiert
    if COLS['length'] not in df.columns:
        df[COLS['length']] = df[COLS['text']].apply(len)

    df_plot = df.copy()
    df_plot['Kategorie'] = df_plot[COLS['target']].map({0: 'Keine Katastrophe', 1: 'Katastrophe'})

    fig = px.histogram(df_plot, x=COLS['length'], color='Kategorie', barmode='overlay', opacity=0.6,
                       color_discrete_map={'Keine Katastrophe': color2, 'Katastrophe': color1},
                       template=PLOT_CONFIG['template'])
    _update_layout(fig, title='Verteilung der Tweet-Länge', xaxis=dict(title="Anzahl Zeichen"),
                   yaxis=dict(title="Häufigkeit"),
                   width=width, height=height)
    if save_path: fig.write_image(save_path)
    fig.show()

    # Vorgabe [2025-10-29]: Unterstützung der Visualisierung durch numerische Tabelle
    print("\n📊 STATISTIKEN ZUR TWEET-LÄNGE:")
    stats = df.groupby(COLS['target'])[COLS['length']].describe().round(2).reset_index()
    stats['Klasse'] = stats[COLS['target']].map({1: 'Katastrophe', 0: 'Normal'})
    display(stats[['Klasse', 'count', 'mean', 'std', 'min', '50%', 'max']])
    del df_plot


def plot_wordclouds(df, save_path=None, col_split=[0.5, 0.5],
                    width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Erstellt Wordclouds für beide Klassen.
    """
    if COLS['clean'] not in df.columns:
        print(f"⚠️ Warnung: '{COLS['clean']}' Spalte fehlt.")
        return

    disaster_text = df[df[COLS['target']] == 1][COLS['clean']].str.cat(sep=' ')
    non_disaster_text = df[df[COLS['target']] == 0][COLS['clean']].str.cat(sep=' ')

    # Dynamische Berechnung der Wordcloud-Dimensionen
    wc_h = int(height * 0.8)
    wc_w1 = int(width * col_split[0] * 0.9)
    wc_w2 = int(width * col_split[1] * 0.9)

    # Wordclouds generieren
    wc_disaster = WordCloud(width=wc_w1, height=wc_h, background_color=PLOT_COLORS['background'],
                            colormap='Reds').generate(disaster_text)
    wc_non = WordCloud(width=wc_w2, height=wc_h, background_color=PLOT_COLORS['background'],
                       colormap='Greens').generate(non_disaster_text)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Katastrophen", "Normal"), column_widths=col_split,
                        horizontal_spacing=0.05)
    fig.add_trace(px.imshow(wc_disaster).data[0], row=1, col=1)
    fig.add_trace(px.imshow(wc_non).data[0], row=1, col=2)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Globalisiertes Layout anwenden
    _update_layout(fig, title="Wordcloud Vergleich", width=width, height=height)

    if save_path: fig.write_image(save_path)
    fig.show()

    # Unterstützung durch numerische Daten (Tabelle)
    print("\n📊 TOP 10 WÖRTER PRO KATEGORIE:")
    top_disaster = Counter(disaster_text.split()).most_common(10)
    top_non = Counter(non_disaster_text.split()).most_common(10)
    df_top = pd.DataFrame(
        {'Katastrophe': [f"{w} ({c})" for w, c in top_disaster], 'Normal': [f"{w} ({c})" for w, c in top_non]})
    display(df_top)
    del disaster_text, non_disaster_text


def plot_ngram_analysis(df, n=2, top_k=20,
                        width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Analysiert und plottet die häufigsten N-Gramme (Wortfolgen).
    n=2 für Bigramme, n=3 für Trigramme.
    """
    if COLS['clean'] not in df.columns:
        print(f"⚠️ Spalte '{COLS['clean']}' nicht gefunden.")
        return

    c_vec = CountVectorizer(ngram_range=(n, n), stop_words='english')

    try:
        ngrams = c_vec.fit_transform(df[COLS['clean']])
        count_values = ngrams.toarray().sum(axis=0)
        vocab = c_vec.vocabulary_
        df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True),
                                columns=['Count', 'N-Gram']).head(top_k)
    except ValueError:
        print(f"⚠️ Nicht genug Daten für {n}-Gramme vorhanden.")
        return

    # Plot Erstellung
    fig = px.bar(df_ngram, x='Count', y='N-Gram', orientation='h', color='Count', color_continuous_scale='Viridis',
                 template=PLOT_CONFIG['template'])

    m_left = int(width * 0.25)

    _update_layout(fig, title=f'Top {top_k} {n}-Gramme (n={n})', yaxis=dict(autorange="reversed"), width=width,
                   height=height)

    fig.update_layout(margin=dict(l=m_left))
    fig.show()

    # Numerische Tabelle zur Unterstützung
    print(f"\n📊 TOP {top_k} {n}-GRAMM DATEN:")
    display(df_ngram)
    del ngrams, df_ngram


def plot_neutral_ngrams(df, n=2, top_n=20,
                        width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Wrapper für plot_ngram_analysis, um Namenskonventionen zu erfüllen
    """
    plot_ngram_analysis(df, n=n, top_k=top_n, width=width, height=height)


def plot_strategic_token_analysis(df, top_n=20,
                                  color1=PLOT_COLORS['disaster'], color2=PLOT_COLORS['no_disaster'],
                                  width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Strategische Analyse: Trennung von Signalwörtern vs. Rauschen.
    Vergleicht die Häufigkeit von Wörtern in beiden Klassen.
    """
    if COLS['clean'] not in df.columns:
        print(f"⚠️ Spalte '{COLS['clean']}' fehlt.")
        return

    def get_top_words(texts, n):
        text_list = texts.dropna().astype(str).tolist()
        all_words = " ".join(text_list).lower().split()
        return Counter(all_words).most_common(n)

    # Daten für beide Klassen sammeln
    df_0 = get_top_words(df[df[COLS['target']] == 0][COLS['clean']], top_n)
    df_1 = get_top_words(df[df[COLS['target']] == 1][COLS['clean']], top_n)
    top_tokens_0 = pd.DataFrame(df_0, columns=["token", "count"])
    top_tokens_0["class"] = "Nicht Katastrophe"
    top_tokens_1 = pd.DataFrame(df_1, columns=["token", "count"])
    top_tokens_1["class"] = "Katastrophe"
    top_tokens = pd.concat([top_tokens_0, top_tokens_1])

    fig = px.bar(top_tokens, x="count", y="token", color="class", barmode="group", orientation="h",
                 color_discrete_map={"Nicht Katastrophe": color2, "Katastrophe": color1},
                 title=f"Strategische Token-Analyse (Top {top_n})", template=PLOT_CONFIG['template'])
    m_left = int(width * 0.18)
    _update_layout(fig, yaxis=dict(autorange="reversed"), width=width, height=height)
    fig.update_layout(margin=dict(l=m_left))
    fig.show()

    # Strategische Analyse Logik
    comparison_df = top_tokens.pivot(index='token', columns='class', values='count').fillna(0)
    overlap = set(top_tokens_0['token']) & set(top_tokens_1['token'])
    noise_candidates = comparison_df[
        (comparison_df['Nicht Katastrophe'] > comparison_df['Katastrophe'] * 1.5)].index.tolist()
    real_signals = comparison_df[(comparison_df['Katastrophe'] > comparison_df['Nicht Katastrophe'] * 2)].index.tolist()

    print("\n🔍 STRATEGISCHE AUSWERTUNG FÜR MODELL-OPTIMIERUNG")
    strat_data = {
        "Typ": ["STOPPWORT-VORSCHLAG", "ECHTE SIGNALE"],
        "Tokens": [", ".join(list(set(noise_candidates) | overlap)), ", ".join(real_signals)]}
    display(pd.DataFrame(strat_data))
    print("\nSIND MANUELL IM NÄCHSTEN CODE EINZUFÜGEN")
    del top_tokens, comparison_df, top_tokens_0, top_tokens_1


def plot_decision_v_shape(model, X, y_true,
                          color1=PLOT_COLORS['disaster'], color2=PLOT_COLORS['no_disaster'],
                          width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    V-Shape Plot zur Visualisierung der Entscheidungssicherheit.
    X-Achse: Wahrscheinlichkeit (0-1)
    Y-Achse: Textlänge
    """
    y_probs = model.predict_proba(X)[:, 1]
    # Textlänge ermitteln
    if 'length' in X.columns:
        lengths = X['length']
    elif 'text' in X.columns:
        lengths = X['text'].apply(len)
    else:
        lengths = np.zeros(len(y_true))

    df_TEMP = pd.DataFrame({'Probability': y_probs, 'Length': lengths, 'Target': y_true})
    df_TEMP['Label'] = df_TEMP['Target'].map({0: 'Keine Katastrophe', 1: 'Katastrophe'})

    # Plot Erstellung
    fig = px.scatter(df_TEMP, x='Probability', y='Length', color='Label', opacity=0.4,
                     color_discrete_map={'Keine Katastrophe': color2, 'Katastrophe': color1},
                     title='Entscheidungs-Visualisierung (Sicherheit vs. Länge)', template=PLOT_CONFIG['template'])
    _update_layout(fig, xaxis=dict(title="Wahrscheinlichkeit (Sicherheit)"), yaxis=dict(title="Textlänge (Zeichen)"),
                   width=width, height=height)
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.7)
    fig.show()

    # Numerische Analyse
    print("\n📊 ANALYSE DER ENTSCHEIDUNGS-ZONEN:")
    # Einteilung in Konfidenz-Zonen
    df_TEMP['Zone'] = pd.cut(df_TEMP['Probability'], bins=[0, 0.4, 0.6, 1.0],
                             labels=['Sicher Normal', 'Unsicher (Grauzone)', 'Sicher Katastrophe'])
    zone_stats = df_TEMP.groupby('Zone').agg({'Length': 'mean', 'Target': 'count'}).rename(
        columns={'Target': 'Anzahl Tweets', 'Length': 'Durchschn. Länge'}).reset_index()
    display(zone_stats)
    del df_TEMP, y_probs


def plot_error_analysis(y_true, y_probs, X_data, threshold=0.5,
                        color1=PLOT_COLORS['disaster'], color2=PLOT_COLORS['no_disaster'],
                        width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Visualisiert Fehler (FN/FP) in einem Scatterplot (Wahrscheinlichkeit vs. Subjektivität).
    X-Achse: Modell-Wahrscheinlichkeit, Y-Achse: Subjektivität.
    """
    # Vorhersage basierend auf Threshold
    y_pred = (y_probs >= threshold).astype(int)

    # Subjektivität berechnen oder holen
    if 'subjectivity' in X_data.columns:
        subjectivity = X_data['subjectivity']
    else:
        col = COLS['clean'] if COLS['clean'] in X_data.columns else COLS['text']
        # Lokale Berechnung falls Spalte fehlt (RAM-schonend via apply)
        subjectivity = X_data[col].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

    df_TEMP = pd.DataFrame(
        {'Wahrscheinlichkeit': y_probs, 'Subjektivität': subjectivity, 'True': y_true, 'Pred': y_pred})
    conditions = [(df_TEMP['True'] == 1) & (df_TEMP['Pred'] == 0), (df_TEMP['True'] == 0) & (df_TEMP['Pred'] == 1)]
    choices = ['Verpasste Katastrophe (FN)', 'Falscher Alarm (FP)']
    df_TEMP['Fehlertyp'] = np.select(conditions, choices, default='Korrekt')
    df_errors = df_TEMP[df_TEMP['Fehlertyp'] != 'Korrekt']

    if df_errors.empty:
        print("✅ Keine Fehler im gewählten Datensatz gefunden!")
        return

    # Plot Erstellung
    fig = px.scatter(df_errors, x='Wahrscheinlichkeit', y='Subjektivität', color='Fehlertyp', opacity=0.7,
                     color_discrete_map={'Verpasste Katastrophe (FN)': color1, 'Falscher Alarm (FP)': color2},
                     title=f'Fehler-Analyse (Threshold: {threshold})', template=PLOT_CONFIG['template'])

    fig.add_vline(x=threshold, line_dash="dash", line_color=PLOT_COLORS['text'],
                  annotation_text="Threshold", annotation_position="top left")

    _update_layout(fig, xaxis=dict(title="Modell-Wahrscheinlichkeit"),
                   yaxis=dict(title="Subjektivität (0=Fakt, 1=Meinung)"),
                   width=width, height=height)
    fig.show()

    # Numerische Unterstützung der Visualisierung
    print("\n📊 FEHLER-STATISTIK & KENNZAHLEN:")
    error_stats = df_errors.groupby('Fehlertyp').agg(
        {'Wahrscheinlichkeit': ['count', 'mean'], 'Subjektivität': 'mean'}).round(3)
    error_stats.columns = ['Anzahl', 'Ø Wahrscheinlichkeit', 'Ø Subjektivität']
    display(error_stats.reset_index())
    del df_TEMP, df_errors


def plot_confusion_matrix_and_hist(y_true, y_pred, y_probs, threshold, col_split=[0.45, 0.55],
                                   color1=PLOT_COLORS['disaster'], color2=PLOT_COLORS['no_disaster'],
                                   width=1000, height=PLOT_CONFIG['height']):
    """
    Kombinierter Plot: Konfusionsmatrix (Links) und Histogramm der Wahrscheinlichkeiten (Rechts).
    """
    # 1. Konfusionsmatrix
    cm = confusion_matrix(y_true, y_pred)
    z_text = [[str(y) for y in x] for x in cm]
    hist_data_1 = y_probs[y_true == 1]
    hist_data_0 = y_probs[y_true == 0]

    # 3. Subplots erstellen
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Matrix (Thresh: {threshold})", "Trennschärfe (Wahrsch.)"),
                        column_widths=col_split, horizontal_spacing=0.1)

    # Linker Plot: Heatmap der Konfusionsmatrix
    fig.add_trace(go.Heatmap(z=cm, x=['Keine Kat', 'Katastrophe'], y=['Keine Kat', 'Katastrophe'],
                             text=z_text, texttemplate="%{text:.2f}", colorscale=PLOT_COLORS['scale_two'],
                             showscale=False), row=1, col=1)
    # Rechter Plot: Histogramm der Verteilungen
    fig.add_trace(go.Histogram(x=hist_data_1, name='Katastrophe', marker_color=color1, opacity=0.6, nbinsx=20), row=1,
                  col=2)
    fig.add_trace(go.Histogram(x=hist_data_0, name='Normal', marker_color=color2, opacity=0.6, nbinsx=20), row=1, col=2)
    fig.add_vline(x=threshold, line_width=2, line_dash="dash", line_color=PLOT_COLORS['text'], row=1, col=2)
    _update_layout(fig, width=width, height=height)
    fig.update_xaxes(title_text="Klasse", row=1, col=1)
    fig.update_xaxes(title_text="Wahrscheinlichkeit", row=1, col=2)
    fig.update_yaxes(title_text="Anzahl", row=1, col=2)
    fig.show()

    # Numerische Tabelle zur Unterstützung
    print(f"\n📊 NUMERISCHE MATRIX & METRIKEN:")
    cm_df = pd.DataFrame(cm, index=['Tatsächlich: Normal', 'Tatsächlich: Katastrophe'],
                         columns=['Vorhergesagt: Normal', 'Vorhergesagt: Katastrophe'])
    display(cm_df)
    del cm, hist_data_1, hist_data_0


def plot_roc_curve(y_true, y_probs,
                   width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Plottet die ROC-Kurve mit AUC-Berechnung.
    """
    # 1. Metriken berechnen
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    df_TEMP = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Threshold': thresholds})

    # 2. Plot Erstellung
    fig = px.area(df_TEMP, x='False Positive Rate', y='True Positive Rate',
                  title=f'ROC Curve (AUC = {roc_auc:.4f})',
                  labels=dict(x='False Positive Rate (1-Spezifität)', y='True Positive Rate (Recall)'),
                  hover_data=['Threshold'], template=PLOT_CONFIG['template'])

    fig.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
    _update_layout(fig, width=width, height=height)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()

    # 3. Numerische Unterstützung
    print("\n📊 ROC-KURVEN STÜTZPUNKTE (Auszug):")
    sample_indices = np.linspace(0, len(df_TEMP) - 1, 10).astype(int)
    display(df_TEMP.iloc[sample_indices].round(4))
    del df_TEMP, fpr, tpr, thresholds


def plot_precision_recall_curve(y_true, y_probs,
                                width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Plottet die Precision-Recall Kurve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    thresholds_adj = np.append(thresholds, 1)
    df_TEMP = pd.DataFrame({'Precision': precision, 'Recall': recall, 'Threshold': thresholds_adj})

    # 2. Plot Erstellung
    fig = px.area(df_TEMP, x='Recall', y='Precision',
                  title='Precision-Recall Curve',
                  labels=dict(x='Recall (Vollständigkeit)', y='Precision (Genauigkeit)'),
                  hover_data=['Threshold'], template=PLOT_CONFIG['template'])

    no_skill = sum(y_true) / len(y_true)
    fig.add_hline(y=no_skill, line_dash="dash", line_color="gray",
                  annotation_text="Zufall (Baseline)", annotation_position="bottom right")

    # Globalisiertes Layout mit prozentualen Margins
    _update_layout(fig, width=width, height=height)
    fig.update_yaxes(range=[0, 1.05])
    fig.update_xaxes(range=[0, 1.05])
    fig.show()

    # 3. Numerische Unterstützung
    print("\n📊 PRECISION-RECALL STÜTZPUNKTE (Auszug):")
    sample_indices = np.linspace(0, len(df_TEMP) - 1, 10).astype(int)
    display(df_TEMP.iloc[sample_indices].sort_values('Threshold').round(4))
    del df_TEMP, precision, recall, thresholds


def plot_feature_importance(model, feature_names, top_n=20,
                            width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Visualisiert die Feature Importance (Koeffizienten oder Gini-Importance)
    """
    importances = None

    # 1. Modell-Extraktion (Pipeline-Check)
    if hasattr(model, 'named_steps'):
        clf = model.named_steps.get('clf')
        if not clf: clf = model.steps[-1][1]
    else:
        clf = model

    # 2. Wichtigkeiten extrahieren (LogReg oder Tree-basiert)
    if hasattr(clf, 'coef_'):
        importances = clf.coef_[0]
    elif hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_

    if importances is not None:
        # INTELLIGENTER FEATURE-NAMEN ABGLEICH
        names_list = list(feature_names)
        diff = len(importances) - len(names_list)

        if diff > 0:
            global df_Cleaning
            if 'df_Cleaning' in globals() and df_Cleaning is not None:
                eng_cols = [c for c in df_Cleaning.columns if c not in [COLS['text'], 'target', 'cleaned_text']]
                names_list.extend(eng_cols[:diff])

            # Falls immer noch Namen fehlen (Fallback)
            while len(names_list) < len(importances):
                names_list.append(f"Eng_Feature_{len(names_list) + 1}")

        # 3. DATEN FÜR PLOT AUFBEREITEN (df_Cleaning Regel nutzen)
        df_plot_feat = pd.DataFrame({'Feature': names_list[:len(importances)], 'Importance': importances})
        df_plot_feat['AbsImportance'] = df_plot_feat['Importance'].abs()
        df_plot_feat = df_plot_feat.sort_values(by='AbsImportance', ascending=False).head(top_n)

        # 4. PLOT-ERSTELLUNG (Mit scale_Tree für Disaster-Einfluss)
        fig = px.bar(df_plot_feat, x='Importance', y='Feature', orientation='h',
                     title=f'Top {top_n} Feature Importance',
                     color='Importance',
                     # Nutzt deine globale Farbskala von Grün nach Rot
                     color_continuous_scale=PLOT_COLORS['scale_Tree'],
                     template=PLOT_CONFIG['template'])

        m_left = int(width * 0.25)
        _update_layout(fig, yaxis=dict(autorange="reversed"), width=width, height=height)
        fig.update_layout(margin=dict(l=m_left))
        fig.show()

        # 5. NUMERISCHE UNTERSTÜTZUNG (Deine Regel vom 29.10.)
        print(f"\n📊 TOP {top_n} FEATURE-WERTE (Numerische Daten):")
        display(df_plot_feat[['Feature', 'Importance']].round(4))

        # RAM-Schutz
        del df_plot_feat
        gc.collect()

    else:
        print("ℹ️ Info: Dieses Modell bietet keine direkte Feature Importance (z.B. KNN oder RBF-SVM).")


def plot_benchmark_results(results_df,
                           width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Vollständiger Scatterplot: Kombiniert Speed (Log-X), F1-Score und Accuracy.
    Nutzt prozentuale Layout-Skalierung und globale Standards.
    """
    df_TEMP = results_df.copy()

    # Plot Erstellung (Blasendiagramm)
    fig = px.scatter(df_TEMP, x="Training Time (s)", y="Recall", color="Model", size="Accuracy",
                     hover_data=["F1-Score", "Precision"],
                     log_x=True, size_max=5, opacity=0.5,
                     title="Modell-Benchmark: Qualität vs. Geschwindigkeit", template=PLOT_CONFIG['template'])

    m_left = int(width * 0.12)
    m_right = int(width * 0.05)
    m_top = int(height * 0.15)
    m_bottom = int(height * 0.12)

    # Globalisiertes Design anwenden
    _update_layout(fig, width=width, height=height, xaxis=dict(title="Trainingszeit (s) - Log Skala"),
                   yaxis=dict(title="Recall (Harmonisches Mittel)"))

    # Spezifische Korrektur für die Legende bei Benchmarks (viele Modelle)
    fig.update_layout(margin=dict(l=m_left, r=m_right, t=m_top, b=m_bottom),
                      legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                                  font=dict(size=min(10, int(width * 0.02)))))
    fig.show()

    # Numerische Datentabelle (Top 10 nach Recall)
    print("\n📊 BENCHMARK DATEN (TOP 10 NACH Recall):")
    display(df_TEMP[['Model', 'Recall', 'F1-Score', 'Accuracy', 'Training Time (s)']].sort_values(by='Recall',
                                                                                                  ascending=False).head(
        10).round(4))
    del df_TEMP


def plot_keras_history(history,
                       width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    Visualisiert den Trainingsverlauf (Loss & Accuracy) von Keras/Deep-Learning Modellen.
    """
    # 1. Daten aus dem History-Objekt extrahieren
    if hasattr(history, 'history'):
        hist_df = pd.DataFrame(history.history)
    else:
        hist_df = pd.DataFrame(history)

    # 2. Plot Erstellung
    fig = px.line(hist_df, y=['loss', 'val_loss', 'accuracy', 'val_accuracy'],
                  title="Trainingsverlauf (Deep Learning)",
                  template=PLOT_CONFIG['template'],
                  color_discrete_map={'loss': PLOT_COLORS['primary'],
                                      'val_loss': PLOT_COLORS['Ladebalken'],
                                      'accuracy': PLOT_COLORS['success'],
                                      'val_accuracy': PLOT_COLORS['error']})

    m_left = int(width * 0.10)
    m_right = int(width * 0.05)
    m_top = int(height * 0.15)
    m_bottom = int(height * 0.15)
    _update_layout(fig, width=width, height=height, xaxis=dict(title="Epoche"), yaxis=dict(title="Metrik-Wert"))

    # Legende optimieren (unten zentriert, da oft 4 Linien)
    fig.update_layout(margin=dict(l=m_left, r=m_right, t=m_top, b=m_bottom),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    fig.show()

    # 3. Numerische Unterstützung
    print("\n📊 FINALE MODELL-METRIKEN (LETZTE EPOCHE):")
    final_metrics = hist_df.iloc[[-1]].round(4).reset_index().rename(columns={'index': 'Epoche'})
    display(final_metrics)
    del hist_df


def plot_confusion_matrix(y_true, y_pred, title="Konfusionsmatrix", labels=['Keine Katastrophe', 'Katastrophe'],
                          width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height'],
                          color_two=PLOT_COLORS['scale_two']):
    """
    Erstellt eine Konfusionsmatrix unter Verwendung der globalen PLOT_CONFIG
    """
    cm = confusion_matrix(y_true, y_pred)

    # Text-Labels für die Quadranten generieren
    z_text = [[str(y) for y in x] for x in cm]

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        annotation_text=z_text,
        colorscale=color_two
    )

    # LAYOUT: Nutzt die übergebenen oder globalen width/height Werte
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=int(height * 0.05))
        ),
        xaxis_title="Vorhergesagt",
        yaxis_title="Tatsächlich",
        width=width,
        height=height,
        paper_bgcolor=PLOT_COLORS['background'],
        plot_bgcolor=PLOT_COLORS['background'],
        template=PLOT_CONFIG['template'],
        margin=dict(l=80, r=80, t=100, b=80)
    )
    fig.show()

    # NUMERISCHE DATENTABELLE (Standard-Regel)
    print(f"\n📊 NUMERISCHE BASISDATEN ({title}):")
    df_metrics = pd.DataFrame(cm, index=[f"Ist: {l}" for l in labels], columns=[f"Vorhergesagt: {l}" for l in labels])
    display(df_metrics)


def plot_evaluate_disaster_model(model, X_test, y_test, feature_cols, threshold=0.35):
    """
    Finale Evaluation mit Custom Threshold und Trennschärfe-Plot.
    [Regel 2025-10-29: Tabelle mit numerischen Daten]
    """
    y_probs = model.predict_proba(X_test[feature_cols])[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    print(f"\n### FINALE ERGEBNISSE (Threshold: {threshold}) ###")

    # 1. Numerischer Report als DataFrame
    report_dict = classification_report(y_test, y_pred,
                                        target_names=['Keine Katastrophe', 'Katastrophe'],
                                        output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose().round(2)
    display(df_report)  # Numerische Tabelle

    # 2. Konfusionsmatrix (Nutzt nlp_utils Standard)
    plot_confusion_matrix(y_test, y_pred, title=f"Ensemble Matrix (T={threshold})")

    # 3. Trennschärfe-Histogramm (Plotly)
    plot_df = pd.DataFrame({
        'Prob': y_probs,
        'Klasse': y_test.map({1: 'Katastrophe', 0: 'Keine Katastrophe'})
    })

    fig = px.histogram(
        plot_df, x='Prob', color='Klasse', barmode='overlay',
        title="<b>Trennschärfe: Signal vs. Rauschen</b>",
        template=PLOT_CONFIG['template'],
        color_discrete_map={'Katastrophe': PLOT_COLORS['success'], 'Keine Katastrophe': PLOT_COLORS['error']}
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color=PLOT_COLORS['text'],
                  annotation_text=f"Threshold {threshold}")
    fig.show()
    del plot_df
    gc.collect()


# ==============================================================================
# 6. MODELLIERUNG & PIPELINES (TRAINING & BENCHMARK)
# ==============================================================================

@contextmanager
def tqdm_joblib(tqdm_object):
    if hasattr(tqdm_object, 'colour'): tqdm_object.colour = TQDM_STYLE['colour']

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.n = tqdm_object.total
        tqdm_object.refresh()
        tqdm_object.close()


def train_advanced_ensemble(X_train, y_train, feature_cols, stop_words):
    """
    Kombiniert Feature Engineering, GridSearch und Hard Example Mining.
    Hält 1 Core frei [Regel Core Reserve Standard].
    """
    print("\n🚀 Starte Ensemble Training & GridSearch (Core Reserve Standard)...")

    # Pipeline Definition
    preprocessor = ColumnTransformer([
        ('tfidf', TfidfVectorizer(stop_words=list(stop_words), ngram_range=(1, 2)), 'lemmatized_text'),
        ('num', StandardScaler(), [c for c in feature_cols if c != 'lemmatized_text'])
    ])

    clf_ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')),
            ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ],
        voting='soft', weights=[2, 1]
    )

    pipeline = Pipeline([('prep', preprocessor), ('clf', clf_ensemble)])
    param_grid = {'clf__lr__C': [0.1, 1, 10], 'clf__rf__n_estimators': [100, 200]}

    # GridSearch mit tqdm Fortschrittsbalken [Regel 2025-11-02]
    total_fits = 6 * 3  # Params * CV
    with tqdm_joblib(tqdm(total=total_fits, desc="Ensemble GridSearch")):
        # n_jobs=-2 lässt exakt einen Core frei für das System
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-2, scoring='recall')
        grid_search.fit(X_train[feature_cols], y_train)

    base_model = grid_search.best_estimator_

    # HARD EXAMPLE MINING (Dein Kernstück)
    print("\n🔍 Refitting: Hard Example Mining (Gewichtung 6.0)...")
    y_probs = base_model.predict_proba(X_train[feature_cols])[:, 1]
    hard_mask = (y_train == 1) & (y_probs < 0.4)

    sample_weights = np.ones(len(y_train))
    sample_weights[hard_mask] = 6.0

    # Finales Fit mit Gewichten
    base_model.fit(X_train[feature_cols], y_train, clf__sample_weight=sample_weights)

    gc.collect()
    return base_model


def find_best_threshold(model, X, y_true):
    """
        Vervollständigt aus defV0.py: Sucht den optimalen Trennpunkt für maximalen F1-Score.
        """
    if not hasattr(model, "predict_proba"):
        return 0.5, f1_score(y_true, model.predict(X), average='macro')

    y_probs = model.predict_proba(X)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)  # Granularität aus defV0
    scores = [f1_score(y_true, (y_probs >= t).astype(int), average='macro', zero_division=0) for t in thresholds]

    ix = np.argmax(scores)
    return thresholds[ix], scores[ix]


def create_submission_file(model, X_test_raw, test_ids, filename=SUBMISSION_PATH):
    if hasattr(model, 'predict_proba'):
        df_test = pd.DataFrame({COLS['text']: X_test_raw})
        df_test[COLS['clean']] = TextCleaner(stopwords.words('english')).transform(df_test[COLS['text']])
        pass
    print(f"Erstelle Submission File: {filename}")


def check_system_performance():
    print("### System-Check ###")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU Kerne: {psutil.cpu_count(logical=True)}")
    print(f"RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU gefunden: {len(gpus)}x {gpus}")
    else:
        print("⚠️ Keine GPU gefunden.")


def run_efficiency_test(X_train, y_train, models, width=1000, height=800):
    print("\n⏱️ Starte Effizienz-Check (Worst-Case Szenario)...")
    X_train['len'] = X_train[COLS['clean']].apply(len)
    longest_indices = X_train.nlargest(5, 'len').index
    X_mini = X_train.loc[longest_indices].copy()
    y_mini = y_train.loc[longest_indices]

    preprocessor = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=100), 'lemmatized_text'),
        ('num', MaxAbsScaler(), [COLS['length']])
    ])

    times = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, model in models:
            try:
                pipeline = Pipeline([('prep', preprocessor), ('clf', model)])
                start = time.time()
                pipeline.fit(X_mini, y_mini)
                y_pred = pipeline.predict(X_mini)
                duration = time.time() - start
                acc = accuracy_score(y_mini, y_pred)
                times.append({'Model': name, 'Time (s)': duration, 'Accuracy': acc})
            except:
                pass

    df_times = pd.DataFrame(times).sort_values('Time (s)')
    fig = px.bar(df_times, x='Model', y='Time (s)', color='Accuracy',
                 title='Vorab-Check: Geschwindigkeit & Genauigkeit (5 längste Texte)',
                 color_continuous_scale='Viridis', text_auto='.4s')
    _update_layout(fig, xaxis=dict(title="Modell"), yaxis=dict(title="Zeit (Sekunden)"), width=width, height=height)
    fig.show()
    return df_times


def run_comprehensive_benchmark(X_train, y_train, X_test, y_test, stop_words):
    print("### START: Benchmark ###")
    check_system_performance()
    X_train_feat = X_train.copy()
    X_test_feat = X_test.copy()
    for df in [X_train_feat, X_test_feat]:
        if COLS['length'] not in df.columns: df[COLS['length']] = df[COLS['text']].apply(len)
        df['lemmatized_text'] = df[COLS['clean']].apply(lemmatize_text)

    feature_cols = ['lemmatized_text', COLS['length']]

    models = [
        ('Dummy', DummyClassifier(strategy='most_frequent')),
        ('LogReg', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', n_jobs=1)),
        ('SGD', SGDClassifier(class_weight='balanced', n_jobs=MODEL_DEFAULTS['n_jobs'])),
        ('Ridge', RidgeClassifier(class_weight='balanced')),
        ('PassiveAggressive', PassiveAggressiveClassifier(class_weight='balanced')),
        ('Perceptron', Perceptron(class_weight='balanced')),
        ('RidgeCV', RidgeClassifierCV(class_weight='balanced')),
        ('Multinomial NB', MultinomialNB()),
        ('Bernoulli NB', BernoulliNB()),
        ('Complement NB', ComplementNB()),
        ('Gaussian NB', GaussianNB()),
        ('Linear SVC', LinearSVC(class_weight='balanced', max_iter=1000, dual='auto')),
        ('SVC (RBF)', SVC(class_weight='balanced', max_iter=1000)),
        ('NuSVC', NuSVC(class_weight='balanced', max_iter=1000)),
        ('Decision Tree', DecisionTreeClassifier(class_weight='balanced')),
        ('Extra Tree', ExtraTreeClassifier(class_weight='balanced')),
        ('Random Forest',
         RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=MODEL_DEFAULTS['n_jobs'])),
        ('Extra Trees',
         ExtraTreesClassifier(n_estimators=100, class_weight='balanced', n_jobs=MODEL_DEFAULTS['n_jobs'])),
        ('Bagging', BaggingClassifier(n_jobs=MODEL_DEFAULTS['n_jobs'])),
        ('AdaBoost', AdaBoostClassifier(n_estimators=50)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100)),
        ('Hist Gradient Boosting', HistGradientBoostingClassifier()),
        ('KNN', KNeighborsClassifier(n_jobs=MODEL_DEFAULTS['n_jobs'])),
        ('Nearest Centroid', NearestCentroid()),
        ('Radius Neighbors', RadiusNeighborsClassifier(radius=100, n_jobs=MODEL_DEFAULTS['n_jobs'])),
        ('LDA', LinearDiscriminantAnalysis()),
        ('QDA', QuadraticDiscriminantAnalysis()),
        ('Gaussian Process', GaussianProcessClassifier(n_jobs=MODEL_DEFAULTS['n_jobs'])),
        ('MLP', MLPClassifier(max_iter=500, early_stopping=True))
    ]

    if xgb: models.append(('XGBoost', xgb.XGBClassifier(n_jobs=MODEL_DEFAULTS['n_jobs'],
                                                        tree_method='gpu_hist' if tf.config.list_physical_devices(
                                                            'GPU') else 'auto')))
    if lgb: models.append(('LightGBM', lgb.LGBMClassifier(n_jobs=MODEL_DEFAULTS['n_jobs'],
                                                          device='gpu' if tf.config.list_physical_devices(
                                                              'GPU') else 'cpu', verbose=-1)))
    if cb: models.append(('CatBoost', cb.CatBoostClassifier(verbose=0, thread_count=MODEL_DEFAULTS['n_jobs'],
                                                            task_type='GPU' if tf.config.list_physical_devices(
                                                                'GPU') else 'CPU')))

    run_efficiency_test(X_train_feat, y_train, models)

    print("\nStarte vollen Benchmark (Warnungen werden unterdrückt)...")
    results = []

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        for name, model in tqdm(models, desc="Benchmarking", **TQDM_STYLE):
            preprocessor = ColumnTransformer([
                ('tfidf', TfidfVectorizer(stop_words=list(stop_words), max_features=5000), 'lemmatized_text'),
                ('num', MaxAbsScaler(), [COLS['length']])
            ])

            if name in ['Gaussian NB', 'QDA', 'LDA', 'Hist Gradient Boosting', 'Gaussian Process']:
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
                    ('clf', model)
                ])
            else:
                pipeline = Pipeline([('prep', preprocessor), ('clf', model)])

            try:
                start_train = time.time()
                pipeline.fit(X_train_feat[feature_cols], y_train)
                train_time = time.time() - start_train

                start_pred = time.time()
                y_pred = pipeline.predict(X_test_feat[feature_cols])
                pred_time = time.time() - start_pred

                results.append({
                    'Model': name,
                    'Training Time (s)': train_time,
                    'Prediction Time (s)': pred_time,
                    'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                    'F1-Score': f1_score(y_test, y_pred, average='macro', zero_division=0),
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0)
                })
            except Exception as e:
                pass

        if len(w) > 0: print(f"\n⚠️ {len(w)} Warnungen wurden während des Benchmarks unterdrückt.")

    results_df = pd.DataFrame(results).sort_values(by=['Recall', 'F1-Score'], ascending=False)
    print(results_df.head(10).to_string(index=False))
    plot_benchmark_results(results_df)

    return results_df


def run_hard_example_mining(model, X_train, y_train, feature_cols):
    """
    Vollständige Logik aus def.py: Zwingt das Modell, aus 'harten' Fehlern zu lernen.
    Integriert sample_weight-Handling für HistGradientBoosting/Voting.
    """
    print("\n⛏️ Hard Example Mining: Analysiere Fehlklassifikationen...")
    # Wahrscheinlichkeiten für die Trainingsdaten berechnen
    y_train_probs = model.predict_proba(X_train[feature_cols])[:, 1]

    # Maske für 'harte' Fälle: Es IST eine Katastrophe, aber Modell sagt 'Sicher' (< 0.4)
    hard_mask = (y_train == 1) & (y_train_probs < 0.4)

    if hard_mask.sum() > 0:
        print(f" -> {hard_mask.sum()} kritische Fehler gefunden. Erhöhe Gewichtung auf Faktor 5.0.")
        sample_weights = np.ones(len(y_train))
        sample_weights[hard_mask] = 5.0  # Einstellung aus def.py übernommen

        # Das Modell wird mit den neuen Gewichten erneut gefittet
        # 'clf__sample_weight' wird genutzt, falls das Modell in einer Pipeline steckt
        try:
            model.fit(X_train[feature_cols], y_train, clf__sample_weight=sample_weights)
        except:
            model.fit(X_train[feature_cols], y_train)  # Fallback
    return model


def run_analysis(X_train, y_train, X_test, y_test, stop_words):
    """
    HAUPT-ANALYSE: Ensemble + Mining + Auto-Tuning.
    Optimiert für Recall, RAM-Schonung und interaktive Plotly-Visualisierung.
    """
    import gc
    from textblob import TextBlob

    print(" START: Analyse Pipeline (Recall-Optimiert)")

    # 1. FEATURE-CHECK & ENGINEERING (Regel 2025-11-21)
    X_train_feat = X_train.copy()
    X_test_feat = X_test.copy()

    def get_local_metrics(text):
        """Sicherheits-Fallback für semantische Features"""
        blob = TextBlob(str(text))
        return pd.Series([blob.sentiment.subjectivity, sum(1 for c in str(text) if c.isupper())])

    for df in [X_train_feat, X_test_feat]:
        print(f"🔍 Prüfe/Generiere Features für Datensatz ({len(df)} Zeilen)...")

        # Nur berechnen, wenn nicht schon durch apply_advanced_engineering vorhanden
        if 'is_llm_emergency' not in df.columns:
            df['llm_score'] = df[COLS['clean']].apply(LLM_0_1_bool)
            df['is_llm_emergency'] = (df['llm_score'] >= 0.7).astype(int)

        if 'disaster_power_score' not in df.columns:
            df['disaster_power_score'] = df[COLS['clean']].apply(calculate_exponential_disaster_score)

        if 'context_trigrams' not in df.columns:
            processed = df[COLS['clean']].apply(process_emojis_and_context)
            df['context_trigrams'] = processed.apply(lambda x: x[1])

        # NameError Fix: Direkte Integration statt externem get_semantic_features
        if 'subjectivity' not in df.columns:
            df[['subjectivity', 'caps_count']] = df[COLS['clean']].apply(get_local_metrics)

        if 'lemmatized_text' not in df.columns:
            df['lemmatized_text'] = df[COLS['clean']].apply(
                lambda x: " ".join([lemmatizer.lemmatize(w) for w in str(x).split()]))

        df['contains_digit'] = df[COLS['clean']].str.contains(r'\d').astype(int)
        df['word_count'] = df[COLS['clean']].apply(lambda x: len(str(x).split()))

    feature_cols = ['lemmatized_text', 'context_trigrams', COLS['length'], 'contains_digit',
                    'word_count', 'subjectivity', 'caps_count', 'disaster_power_score',
                    'is_noise_flag', 'is_llm_emergency']

    # 2. PIPELINE AUFBAU (1 Core Frei via MODEL_DEFAULTS['n_jobs'])
    preprocessor = ColumnTransformer([
        ('tfidf_main', TfidfVectorizer(stop_words=list(stop_words), ngram_range=(1, 2)), 'lemmatized_text'),
        ('tfidf_context', TfidfVectorizer(ngram_range=(1, 3)), 'context_trigrams'),
        ('num', StandardScaler(), [COLS['length'], 'contains_digit', 'word_count', 'subjectivity',
                                   'caps_count', 'disaster_power_score', 'is_noise_flag']),
        ('llm_signal', StandardScaler(with_mean=False), ['is_llm_emergency'])
    ])

    clf_ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')),
            ('rf', RandomForestClassifier(random_state=42, n_jobs=1, class_weight='balanced')),
            ('hgb', HistGradientBoostingClassifier(random_state=42))
        ], voting='soft', weights=[2, 1, 2]
    )

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
        ('clf', clf_ensemble)
    ])

    # 3. AUTO-TUNING
    param_grid = {
        'clf__lr__C': [1, 10],
        'clf__hgb__learning_rate': [0.1, 0.05]
    }

    # tqdm Progress Bar Integration
    total_fits = len(param_grid['clf__lr__C']) * len(param_grid['clf__hgb__learning_rate']) * 3
    with tqdm_joblib(tqdm(total=total_fits, desc="GridSearch Ensemble", **TQDM_STYLE)):
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=RESERVED_CORES, scoring='recall')
        grid_search.fit(X_train_feat[feature_cols], y_train)

    best_model = grid_search.best_estimator_

    # 4. HARD EXAMPLE MINING (Gewicht 5.0 oder 6.0 nach Standard)
    y_train_probs = best_model.predict_proba(X_train_feat[feature_cols])[:, 1]
    hard_mask = (y_train == 1) & (y_train_probs < 0.4)

    if hard_mask.sum() > 0:
        print(f"⛏️ Mining: {hard_mask.sum()} harte Fälle gefunden. Refit...")
        sample_weights = np.ones(len(y_train))
        sample_weights[hard_mask] = 6.0  # Mining Standard
        best_model.fit(X_train_feat[feature_cols], y_train, clf__sample_weight=sample_weights)

    # 5. THRESHOLD & EVALUATION (Plotly)
    best_thresh, _ = find_best_threshold(best_model, X_test_feat[feature_cols], y_test)
    y_probs = best_model.predict_proba(X_test_feat[feature_cols])[:, 1]
    y_pred = (y_probs >= best_thresh).astype(int)

    # Numerische Validierung (Regel 2025-10-29)
    print("\n📊 FINALE METRIKEN:")
    display(pd.DataFrame([classification_report(y_test, y_pred, output_dict=True)['macro avg']]))

    # Plotly Visualisierung (Interaktiv)
    plot_confusion_matrix(y_test, y_pred)  # Deine Plotly-Funktion

    gc.collect()
    return best_model


def run_llm_direct_evaluation(X_test, y_test):
    """
    NEU: Direkte Auswertung nur basierend auf dem LLM-Score.
    """
    print("### START: LLM Direct Evaluation ###")
    X_test_feat = X_test.copy()
    X_test_feat['llm_score'] = X_test_feat[COLS['clean']].apply(LLM_0_1_bool)

    # Threshold Optimierung für LLM
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = [f1_score(y_test, (X_test_feat['llm_score'] >= t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = thresholds[np.argmax(scores)]

    y_pred = (X_test_feat['llm_score'] >= best_thresh).astype(int)

    print(f"Optimaler LLM-Threshold: {best_thresh:.2f}")
    print(classification_report(y_test, y_pred, target_names=['Keine', 'Katastrophe']))
    plot_confusion_matrix_and_hist(y_test, y_pred, X_test_feat['llm_score'], best_thresh)


def run_bert_training(X_train_raw, y_train, X_test_raw, y_test):
    """
    Trainiert ein DistilBERT-Modell mit KerasNLP.
    """
    print("### START: BERT Pipeline ###")
    check_system_performance()
    if os.path.exists(KERAS_MODEL_PATH):
        classifier = tf.keras.models.load_model(KERAS_MODEL_PATH)
    else:
        classifier = keras_nlp.models.DistilBertClassifier.from_preset("distil_bert_base_en_uncased", num_classes=2)
        classifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           optimizer=tf.keras.optimizers.Adam(3e-5), metrics=["accuracy"])

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        history = classifier.fit(x=X_train_raw, y=y_train, batch_size=16, epochs=5,
                                 validation_data=(X_test_raw, y_test),
                                 callbacks=[TqdmCallback(verbose=1), EarlyStopping(monitor='val_loss', patience=2)],
                                 class_weight=dict(enumerate(class_weights)))
        classifier.save(KERAS_MODEL_PATH)
        plot_keras_history(history)

    y_pred_logits = classifier.predict(X_test_raw, verbose=0)
    y_probs = tf.nn.softmax(y_pred_logits)[:, 1].numpy()
    y_pred = np.argmax(y_pred_logits, axis=1)

    print(classification_report(y_test, y_pred))
    plot_confusion_matrix_and_hist(y_test, y_pred, y_probs, 0.5)
    plot_roc_curve(y_test, y_probs)
    return classifier


def run_master_study_ensemble(X_train_raw, X_test_raw, y_train, y_test, stop_words,width=PLOT_CONFIG['width'], height=PLOT_CONFIG['height']):
    """
    MASTER-STUDIE: Signal-Analyse, Ensemble-Training mit Mining (6.0) und Plotly-Evaluation.
    """
    # SCHRITT 1: SIGNAL- UND RAUSCH-ANALYSE
    print(f"STUDIE TEIL 1: SIGNAL- UND RAUSCH-ANALYSE")
    df_TEMP = X_train_raw[[COLS['clean']]].copy()

    words_disaster = " ".join(df_TEMP.loc[y_train == 1, COLS['clean']]).split()
    words_normal = " ".join(df_TEMP.loc[y_train == 0, COLS['clean']]).split()
    freq_disaster, freq_normal = Counter(words_disaster), Counter(words_normal)

    relevance_data = []
    threshold = 5 if len(X_train_raw) > 300 else 1

    for word in set(words_disaster) | set(words_normal):
        f_dis, f_norm = freq_disaster[word], freq_normal[word]
        if (f_dis + f_norm) >= threshold:
            relevance_data.append({
                'Wort': word,
                'Relevanz': round(f_dis / (f_norm + 1), 2),
                'Total': f_dis + f_norm
            })

    df_study = pd.DataFrame(relevance_data).sort_values(by='Relevanz', ascending=False)
    print(f"\n📊 TOP 10 SIGNALWÖRTER (Threshold >= {threshold}):")
    display(df_study.head(10))

    gc.collect()

    # SCHRITT 2: FEATURE ENGINEERING (REDUNDANZ-FIX)
    print(f"\nSCHRITT 2: FEATURE PREPARATION & LEMMATISIERUNG")
    for df_split in [X_train_raw, X_test_raw]:
        df_split['contains_digit'] = df_split[COLS['clean']].str.contains(r'\d').astype(int)
        # Lemmatisierung nutzt den globalen Spaltennamen
        df_split[COLS['lemmatized']] = df_split[COLS['clean']].apply(
            lambda x: " ".join([lemmatizer.lemmatize(w) for w in str(x).split()]))

    # Feature-Liste konsolidiert
    feature_cols = [COLS['lemmatized'], COLS['length'], 'contains_digit', 'word_count', 'subjectivity', 'caps_count']

    # SCHRITT 3: ENSEMBLE MODELLIERUNG (CORE RESERVE: n_jobs=-2)
    print(f"\n🚀 Starte Ensemble Training (Core-Reserve: {RESERVED_CORES} Cores)...")

    preprocessor = ColumnTransformer([
        ('tfidf', TfidfVectorizer(stop_words=list(stop_words), ngram_range=(1, 2)), COLS['lemmatized']),
        ('num', StandardScaler(), [c for c in feature_cols if c != COLS['lemmatized']])
    ])

    clf_ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')),
            ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ], voting='soft', weights=[2, 1]
    )

    pipeline = Pipeline([('prep', preprocessor), ('clf', clf_ensemble)])
    param_grid = {'clf__lr__C': [0.1, 1, 10], 'clf__rf__n_estimators': [100, 200]}

    total_fits = 18
    with tqdm_joblib(tqdm(total=total_fits, desc="Ensemble GridSearch", **TQDM_STYLE)):
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=RESERVED_CORES, scoring='recall')
        grid_search.fit(X_train_raw[feature_cols], y_train)

    base_model = grid_search.best_estimator_

    # SCHRITT 4: HARD EXAMPLE MINING (GEWICHT 6.0)
    print(f"\n🔍 SCHRITT 4: HARD EXAMPLE MINING (Refit mit Gewicht 6.0)...")
    y_probs_train = base_model.predict_proba(X_train_raw[feature_cols])[:, 1]
    hard_mask = (y_train == 1) & (y_probs_train < 0.4)

    sample_weights = np.ones(len(y_train))
    sample_weights[hard_mask] = 6.0

    # Refit unter Berücksichtigung der Gewichtung
    base_model.fit(X_train_raw[feature_cols], y_train, clf__sample_weight=sample_weights)

    # SCHRITT 5: FINALE EVALUATION (GLOBAL COLORS & TERMS)
    print(f"\n### STUDIE TEIL 3: FINALE ERGEBNISSE (Interaktiv) ###")
    y_probs = base_model.predict_proba(X_test_raw[feature_cols])[:, 1]
    y_pred = (y_probs >= 0.35).astype(int)

    # 1. Numerische Tabelle
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2).reset_index()
    display(report_df)

    # 2. Interaktive Confusion Matrix (Nutzt PLOT_COLORS['scale_two'])
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ff.create_annotated_heatmap(
        z=cm, x=['Kein Disaster', 'Disaster'], y=['Echt: 0', 'Echt: 1'],
        colorscale=PLOT_COLORS['scale_two'], showscale=True
    )
    fig_cm.update_layout(
        title="<b>Interaktive Confusion Matrix</b>",
        width=width, height=height,
        template=PLOT_CONFIG['template']
    )
    fig_cm.show()

    # 3. Interaktives Histogramm
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=y_probs[y_test == 1], name='Disaster',
        marker_color=PLOT_COLORS['disaster'], opacity=0.75
    ))
    fig_hist.add_trace(go.Histogram(
        x=y_probs[y_test == 0], name='Kein Disaster',
        marker_color=PLOT_COLORS['no_disaster'], opacity=0.75
    ))

    fig_hist.add_vline(x=0.35, line_dash="dash", line_color=PLOT_COLORS['text'], annotation_text="Threshold 0.35")
    fig_hist.update_layout(
        title="<b>Wahrscheinlichkeitsverteilung (Mining 6.0)</b>",
        barmode='overlay',
        paper_bgcolor=PLOT_COLORS['background'],  # Äußerer Rahmen
        plot_bgcolor=PLOT_COLORS['background'],
        template=PLOT_CONFIG['template'],
        xaxis_title="Score",
        yaxis_title="Anzahl",
        xaxis=dict(gridcolor=PLOT_COLORS['grid']),
        yaxis=dict(gridcolor=PLOT_COLORS['grid'])
    )
    fig_hist.show()

    gc.collect()
    return base_model