# Retail Demand Forecast für die Region Guayas

## 1. Projektziel

Dieses Projekt zielt darauf ab, die Verkaufszahlen für Einzelhandelsprodukte in der Region Guayas, Ecuador, vorherzusagen. Das Kernstück ist eine interaktive Webanwendung, die es Bedarfsplanern ermöglicht, Prognosen für spezifische Shop-Artikel-Kombinationen zu visualisieren und zu analysieren.

Das trainierte Modell ist ein **XGBoost Regressor**, der aufgrund seiner hohen Leistung und Geschwindigkeit im Vergleich zu einem LSTM-Basismodell ausgewählt wurde.

## 2. Leistung des Modells

Das optimierte XGBoost-Modell wurde auf einem chronologischen Test-Set (März 2014) evaluiert und erzielte folgende Metriken:

- **Mean Absolute Error (MAE):** 0.6587
- **Root Mean Squared Error (RMSE):** 0.8103

Diese Metriken basieren auf log-transformierten Verkaufszahlen (`log1p`). Der MAE von ~0.66 bedeutet, dass die Prognose im Durchschnitt um diesen Wert von den tatsächlichen Log-Verkäufen abweicht.

## 3. App-Übersicht

Die Streamlit-Anwendung bietet folgende Funktionen:

- **Interaktive Filter:** Benutzer können über eine Seitenleiste einen spezifischen Shop (`store_nbr`) und einen Artikel (`item_nbr`) auswählen.
- **Visueller Vergleich:** Ein interaktives Liniendiagramm stellt die tatsächlichen Verkaufszahlen den prognostizierten Werten für den Zeitraum Januar bis März 2014 gegenüber.
- **Rohdatenansicht:** Eine Tabelle unterhalb des Diagramms zeigt die genauen numerischen Werte für tatsächliche und prognostizierte Verkäufe.

### Screenshot

*(Hier könntest du einen Screenshot deiner laufenden App einfügen)*

## 4. Setup & Ausführung

Folge diesen Schritten, um die Anwendung lokal auszuführen:

1.  **Repository klonen:**
    ```bash
    git clone <dein-repository-link>
    cd retail_demand_analysis
    ```

2.  **Abhängigkeiten installieren:**
    Erstelle eine virtuelle Umgebung und installiere die notwendigen Pakete.
    ```bash
    python -m venv venv
    source venv/bin/activate  # Auf Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Streamlit-App starten:**
    Führe den folgenden Befehl im Hauptverzeichnis des Projekts aus:
    ```bash
    streamlit run app/main.py
    ```

4.  Die Anwendung wird in deinem Browser unter `http://localhost:8501` geöffnet.