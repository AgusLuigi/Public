# Muster: config_farben.py

# Dient als zentrale Konfigurationsdatei für Farbpaletten und Plot-Stile
# in allen Jupyter Notebooks und der finalen Streamlit-App.

# -----------------------------------------------------------------------------
# 1. SEQUENZ-PALETTE (Liste für Linienplots und Farbzyklen)
# -----------------------------------------------------------------------------
SEQUENZ_PALETTE = [
    "#0077B6",  # Dunkelblau (Ist-Werte/Daten)
    "#D90429",  # Rot (Prognose/Fehler)
    "#06D6A0",  # Türkis (Vergleichsmodell 1)
    "#FFC300",  # Orange (Trend/Saisonalität)
    "#6A057F",  # Violett
    "#A1C4FD"   # Hellblau
]

# -----------------------------------------------------------------------------
# 2. FARBPALETTE (Dictionary für spezifische, benannte Farben)
# -----------------------------------------------------------------------------
FARBPALETTE = {
    "Ist_Wert": "#0077B6",
    "Prognose": "#D90429",
    "Gut": "#06D6A0",      # Für positive Metriken
    "Mittel": "#FFC300",   # Für Warnungen
    "Schlecht": "#A1C4FD"  # Für kritische Fehler
}

# -----------------------------------------------------------------------------
# 3. Optional: Globaler Seaborn/Matplotlib-Stil
# -----------------------------------------------------------------------------
SEABORN_STYLE = "whitegrid" 

# Hinweis: Für die Anwendung in Notebooks verwenden Sie:
# from config.config_farben import SEQUENZ_PALETTE
# import seaborn as sns
# sns.set_palette(SEQUENZ_PALETTE)
