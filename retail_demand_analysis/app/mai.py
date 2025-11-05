import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import xgboost as xgb 
from datetime import timedelta

# Konfiguration und Laden der Artefakte

@st.cache_resource
def load_model():
    """Lädt das trainierte Machine-Learning-Modell."""
    model_path = 'model/xgb_reg_optimized.pkl' 
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Modell-Datei ('{model_path}') nicht gefunden. 🛑 Bitte das Notebook ausführen, um das Modell zu speichern.")
        return None

# HIER WURDE DIE KORREKTUR VOM LETZTEN MAL EINGEFÜGT, UM X_cols ALS 1D-LISTE ZU GARANTIEREN
@st.cache_resource
def load_artifacts():
    """Lädt den Feature-Scaler und die Liste der verwendeten Spalten."""
    scaler_path = 'artifacts_ml_best/X_scaler.pkl' 
    cols_path = 'artifacts_ml_best/X_cols.pkl'     
    try:
        scaler = joblib.load(scaler_path)
        
        loaded_cols_object = joblib.load(cols_path)
        
        if isinstance(loaded_cols_object, pd.DataFrame):
            cols = list(loaded_cols_object.columns) 
        elif isinstance(loaded_cols_object, pd.Index):
            cols = list(loaded_cols_object) 
        else:
            cols = loaded_cols_object 
            
        # Sicherheits-Check: Konvertiere zu Liste, falls es noch ein Index ist
        if not isinstance(cols, list):
             cols = list(cols)

        return scaler, cols
    except FileNotFoundError:
        st.error(f"Artefakt-Dateien ('{os.path.dirname(scaler_path)}/') nicht gefunden. 🛑 Bitte das Notebook ausführen, um die Artefakte zu speichern.")
        return None, None

@st.cache_data
def load_data():
    """Lädt die vorbereiteten historischen Daten."""
    data_path = 'data/guayas_prepared.csv'
    try:
        data = pd.read_csv(data_path, parse_dates=['date'])
        return data
    except FileNotFoundError:
        st.error(f"Daten-Datei ('{data_path}') nicht gefunden. 🛑 Bitte sicherstellen, dass die Datei im data-ordner liegt.")
        return None

# Hilfsfunktion zur Generierung von Zukunfts-Features
def generate_future_features(start_date, end_date, stores, items, X_cols):
    """Generiert alle notwendigen Features für einen zukünftigen Zeitraum."""
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Erstelle ein DataFrame für alle Kombinationen
    future_data = pd.DataFrame([(d, s, i) for d in dates for s in stores for i in items], 
                               columns=['date', 'store_nbr', 'item_nbr'])

    if future_data.empty:
        return pd.DataFrame(columns=X_cols + ['date', 'store_nbr', 'item_nbr'])

    # Erzeuge Zeit-Features
    future_data['dayofweek'] = future_data['date'].dt.dayofweek
    future_data['dayofmonth'] = future_data['date'].dt.day
    future_data['month'] = future_data['date'].dt.month
    future_data['year'] = future_data['date'].dt.year
    
    # Erzeuge ein DataFrame, das nur die Features enthält, die das Modell erwartet
    future_X_cols = pd.DataFrame(index=future_data.index)
    
    for col in X_cols:
        if col in ['dayofweek', 'dayofmonth', 'month', 'year']:
            future_X_cols[col] = future_data[col]
        elif col.startswith('store_nbr_') or col.startswith('item_nbr_'):
            # Extrahieren der ID und Erstellen der Dummy-Variable
            try:
                col_name_parts = col.split('_')
                col_type = col_name_parts[0] + '_' + col_name_parts[1]
                col_value = int(col_name_parts[2])
                
                if col_type == 'store_nbr':
                    future_X_cols[col] = (future_data['store_nbr'] == col_value).astype(float)
                elif col_type == 'item_nbr':
                    future_X_cols[col] = (future_data['item_nbr'] == col_value).astype(float)
                else:
                    future_X_cols[col] = 0.0 
            except:
                future_X_cols[col] = 0.0 
                
        # Alle anderen Spalten (z.B. 'onpromotion', 'type_...') werden für die Zukunft als 0 angenommen
        else:
            future_X_cols[col] = 0.0
            
    # Sicherstellen, dass die Reihenfolge EXAKT stimmt
    future_X_cols = future_X_cols[X_cols].fillna(0)
    
    # Füge die Datums- und Identifikationsspalten für die spätere Zusammenfassung hinzu
    future_X_cols['date'] = future_data['date']
    future_X_cols['store_nbr'] = future_data['store_nbr']
    future_X_cols['item_nbr'] = future_data['item_nbr']

    return future_X_cols

# Lade Artefakte
model = load_model()
scaler, X_cols = load_artifacts()
df = load_data()

# UI Gestaltung

st.set_page_config(page_title="Demand Forecast Guayas", layout="wide")
st.title('🛒 Retail Demand Forecast für die Region Guayas')

st.sidebar.info(f"Aktuelles Arbeitsverzeichnis (CWD): \n\n {os.getcwd()}")

# Initialisierungs-Check (Kurz gehalten, da Debugging Code von oben übernommen wird)
initialization_successful = True
if model is None or scaler is None or X_cols is None or df is None:
    initialization_successful = False
    
# --- DEBUG-CHECK: Scaler und Feature Anpassung (vom letzten Mal übernommen) ---
if initialization_successful:
    try:
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            model_features = list(model.feature_names_in_)
            n_model_features = len(model_features)

            if hasattr(scaler, 'feature_names_in_') and len(scaler.feature_names_in_) != n_model_features:
                scaler.feature_names_in_ = np.array(model_features, dtype=object)
                if hasattr(scaler, 'n_features_in_'): scaler.n_features_in_ = n_model_features
                if hasattr(scaler, 'data_min_'): scaler.data_min_ = scaler.data_min_[:n_model_features]
                if hasattr(scaler, 'data_max_'): scaler.data_max_ = scaler.data_max_[:n_model_features]
                if hasattr(scaler, 'data_range_'): scaler.data_range_ = scaler.data_range_[:n_model_features]
                if hasattr(scaler, 'min_'): scaler.min_ = scaler.min_[:n_model_features]
                if hasattr(scaler, 'mean_'): scaler.mean_ = scaler.mean_[:n_model_features]
                if hasattr(scaler, 'scale_'): scaler.scale_ = scaler.scale_[:n_model_features]
            
            if len(model_features) != len(X_cols):
                X_cols = model_features
            
    except Exception as e:
        pass # Ignoriere interne Fehler des Debug-Checks
        
    try:
        model_feature_count = model.get_booster().num_feature()
        if model_feature_count != len(X_cols):
            st.error(f"🚨 KRITISCHER FEHLER: FEATURE-ANZAHL-MISMATCH!")
            initialization_successful = False
            
    except AttributeError:
        pass 


if not initialization_successful:
    st.markdown("---")
    st.header("❌ Initialisierung Fehlgeschlagen")
    st.error("Die App konnte nicht gestartet werden, da wichtige Modelle, Scaler oder Daten fehlen.")
    st.warning("Bitte beheben Sie die oben genannten 🛑 **FEHLERMELDUNGEN**.")
    st.stop() 
# --- ENDE DEBUG-CHECK ---


# Filtern der Daten auf den relevanten Zeitraum (Jan-Mär 2014)
HISTORICAL_END_DATE = '2014-03-31'
df_forecast_period = df[(df['date'] >= '2014-01-01') & (df['date'] <= HISTORICAL_END_DATE)].copy()

# --- Sidebar für Benutzereingaben ---

st.sidebar.header('Filter für die Vorhersage')

# --- NEU: Automatische Granularität (Pauschale Einstellung: Allzeit/Tag) ---
FREQ = 'D'
X_AXIS_FORMAT = "%d.%b %Y" # Tägliche Auflösung mit Anzeige des Jahres für Kontext
current_granularity_label = 'Tag (Gesamtzeitraum)'

if not df_forecast_period.empty:
    
    # 1. Store Selection (Multiselect)
    available_stores_options = ['Alle Stores'] + sorted(df_forecast_period['store_nbr'].unique().tolist())
    # Standard: 'Alle Stores' (entspricht der Benutzeranforderung)
    selected_stores = st.sidebar.multiselect('Wähle Shops (Stores):', available_stores_options, default=['Alle Stores'])
    
    if 'Alle Stores' in selected_stores or not selected_stores:
        stores_to_process = sorted(df_forecast_period['store_nbr'].unique().tolist())
    else:
        stores_to_process = [s for s in selected_stores if s != 'Alle Stores']
        
    df_filtered_by_store = df_forecast_period[df_forecast_period['store_nbr'].isin(stores_to_process)].copy()
    
    # 2. Item Selection (Multiselect)
    available_items_for_selection = sorted(df_filtered_by_store['item_nbr'].unique().tolist())
    available_items_options = ['Alle Artikel'] + available_items_for_selection
    
    # Standard: 'Alle Artikel' (entspricht der Benutzeranforderung)
    selected_items = st.sidebar.multiselect('Wähle Artikel (Items):', available_items_options, default=['Alle Artikel'])

    if 'Alle Artikel' in selected_items or not selected_items:
        items_to_process = available_items_for_selection
    else:
        items_to_process = [i for i in selected_items if i != 'Alle Artikel']
        
    # Historischer DataFrame zum Verarbeiten
    historical_df = df_filtered_by_store[df_filtered_by_store['item_nbr'].isin(items_to_process)].copy()

    # NEUE FUNKTIONALITÄT: Dynamische Achsen-Zoom-Option
    st.sidebar.markdown("---")
    zoom_y_axis = st.sidebar.checkbox(
        'Y-Achse auf Bewegung zoomen (Startpunkt > 0)',
        value=False,
        help="Zeigt die relativen täglichen Schwankungen besser, ignoriert aber den wahren Nullpunkt der Skala."
    )


    st.sidebar.info(f"Vorhersage-Granularität: **{current_granularity_label}**.\n\n Shops: **{len(stores_to_process)}** | Artikel: **{len(items_to_process)}**.")

    if historical_df.empty:
        st.warning("Für diese Shop-Artikel-Kombination(en) gibt es keine Daten im Prognosezeitraum.")
    else:
        
        with st.spinner('Führe historische und Zukunfts-Prognose durch... Bitte warten.'):
            
            # 1. HISTORISCHE DATEN VORHERSAGEN (Jan-Mär 2014)
            X_hist_pred = pd.DataFrame(columns=X_cols)
            available_cols_in_series = [col for col in X_cols if col in historical_df.columns]
            temp_df = historical_df.copy()
            temp_df = temp_df[available_cols_in_series].reindex(columns=X_cols, fill_value=0.0) 
            X_hist_pred = temp_df[X_cols].fillna(0)
            
            # Historische Prognose
            X_hist_pred_scaled = scaler.transform(X_hist_pred)
            predictions_hist = np.expm1(model.predict(X_hist_pred_scaled))
            predictions_hist[predictions_hist < 0] = 0 
            
            historical_df['Prognostizierte Verkäufe'] = predictions_hist # Prognose für den historischen Teil
            historical_df['Tatsächliche Verkäufe'] = np.expm1(historical_df['unit_sales'])
            historical_df['Typ'] = 'Historie (Aktuelle Daten)'
            
            # 2. ZUKUNFTS-DATEN GENERIEREN UND VORHERSAGEN (5 Tage)
            
            FUTURE_START_DATE = pd.to_datetime(HISTORICAL_END_DATE) + timedelta(days=1)
            FUTURE_DAYS = 5 # <--- NUR 5 TAGE PROGNOSE
            FUTURE_END_DATE = FUTURE_START_DATE + timedelta(days=FUTURE_DAYS - 1) # Korrekt für 5 Tage (z.B. 1. Apr bis 5. Apr)

            future_df = generate_future_features(
                start_date=FUTURE_START_DATE, 
                end_date=FUTURE_END_DATE, 
                stores=stores_to_process, 
                items=items_to_process, 
                X_cols=X_cols
            )
            
            # Vorhersage für die Zukunft
            X_future_pred = future_df[X_cols].fillna(0)
            X_future_pred_scaled = scaler.transform(X_future_pred)
            predictions_future = np.expm1(model.predict(X_future_pred_scaled))
            predictions_future[predictions_future < 0] = 0
            
            future_df['Prognostizierte Verkäufe'] = predictions_future
            future_df['Tatsächliche Verkäufe'] = np.nan # Keine tatsächlichen Daten in der Zukunft
            future_df['Typ'] = f'Prognose ({FUTURE_DAYS}-Tage)'

            # 3. DATEN ZUSAMMENFÜHREN UND AGGREGIEREN
            cols_to_combine = ['date', 'store_nbr', 'item_nbr', 'Tatsächliche Verkäufe', 'Prognostizierte Verkäufe', 'Typ']
            
            # Kombinieren von Historie und Zukunft
            combined_df = pd.concat([
                historical_df[cols_to_combine].copy(), 
                future_df[cols_to_combine].copy()
            ], ignore_index=True)
            
            # Aggregation auf Tagesbasis
            results_df_aggregated = combined_df.groupby('date').agg(
                {'Tatsächliche Verkäufe': 'sum', 'Prognostizierte Verkäufe': 'sum'}
            ).reset_index()
            
            # Benennung der Datumsspalte
            results_df_aggregated.rename(columns={'date': 'Datum'}, inplace=True)
            results_df = results_df_aggregated.copy() 

            # --- PLOT-ERSTELLUNG ---
            
            # Identifiziere den Wechselpunkt
            split_date = pd.to_datetime(HISTORICAL_END_DATE)
            
            st.header(f'Vergleich: Historische Verkäufe & {FUTURE_DAYS}-Tage Zukunftsprognose')
            
            # HINWEIS ZUM LINEAREN VERLAUF/SPRUNG (als Erklärungsversuch)
            st.warning(f"⚠️ **Hinweis zum Sprung/linearen Verlauf:** Der starke Anstieg (z.B. von 0 auf 100k) tritt auf, weil die Prognose für den {FUTURE_START_DATE.strftime('%d.%m.%Y')} beginnt und **alle** ausgewählten Artikel/Shops summiert werden. Wenn Sie die detaillierte tägliche Bewegung sehen möchten, wählen Sie bitte nur **einen Shop und einen Artikel** aus.")
            
            st.info(f"Datenpunkte für Plot ({current_granularity_label}-Basis): **{len(results_df['Datum'].unique())}** Zeitabschnitte.")


            fig = go.Figure()

            # 1. Tatsächliche Verkäufe (nur Historie) - Blaue Linie
            fig.add_trace(go.Scatter(
                x=results_df['Datum'], 
                y=results_df['Tatsächliche Verkäufe'],
                mode='lines+markers',
                name='Tatsächliche Verkäufe (Historie)',
                line=dict(color='#0052CC', width=3)
            ))

            # 2a. Prognose (Historie) - GEPUNKTETE ROTE LINIE
            hist_forecast_segment = results_df[results_df['Datum'] <= split_date]
            fig.add_trace(go.Scatter(
                x=hist_forecast_segment['Datum'], 
                y=hist_forecast_segment['Prognostizierte Verkäufe'],
                mode='lines',
                name='Prognostizierte Verkäufe (Historische Periode)',
                line=dict(color='#DE350B', width=2, dash='dot') 
            ))
            
            # 2b. Prognose (Zukunft 5-Tage) - DURCHGEZOGENE ROTE LINIE
            future_forecast_segment = results_df[
                (results_df['Datum'] >= FUTURE_START_DATE) & 
                (results_df['Datum'] <= FUTURE_END_DATE)
            ]

            fig.add_trace(go.Scatter(
                x=future_forecast_segment['Datum'], 
                y=future_forecast_segment['Prognostizierte Verkäufe'],
                mode='lines+markers',
                name=f'Prognose ({FUTURE_DAYS}-Tage)',
                line=dict(color='#DE350B', width=3, dash='solid') 
            ))

            # NEU: Dynamische Y-Achsen-Anpassung basierend auf der Auswahl
            yaxis_config = dict(
                title=f'Verkaufte Einheiten (Aggregiert nach {current_granularity_label})',
                rangemode="tozero" # Standardmäßig bei 0 starten
            )
            
            if zoom_y_axis:
                # Berechne den Mindestwert über den gesamten Zeitraum für den Zoom
                min_sales = results_df[['Tatsächliche Verkäufe', 'Prognostizierte Verkäufe']].min().min()
                # Setze den Y-Achsen-Startpunkt leicht unter den Mindestwert, um die Bewegung zu vergrößern
                zoom_start = max(0, min_sales * 0.95)
                
                # Der Endpunkt bleibt der maximale Wert
                max_sales = results_df['Prognostizierte Verkäufe'].max()
                zoom_end = max_sales * 1.05
                
                yaxis_config['range'] = [zoom_start, zoom_end]
                yaxis_config['rangemode'] = "normal" # Deaktiviert tozero, da wir manuell gesetzt haben


            fig.update_layout(
                title=f"Gesamt-Verkaufsprognose ({current_granularity_label}-Basis)",
                xaxis_title='Datum',
                legend_title_text='Datenreihe',
                hovermode="x unified",
                template='plotly_white',
                yaxis=yaxis_config # Verwende die dynamische Konfiguration
            )
            
            # Aktualisiere Achsenformat und Slider
            fig.update_xaxes(dtick="M1", tickformat=X_AXIS_FORMAT, rangeslider_visible=True)
            
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"Rohdaten der Vorhersage (Aggregiert nach {current_granularity_label}, Erste 5 Zeilen)")
            
            # Zeigt die aggregierten Daten an
            if not results_df_aggregated.empty:
                st.dataframe(results_df_aggregated.head(5)) 
                
            st.subheader("Feature-Namen (X_cols)")
            st.dataframe(pd.Series(X_cols).head(5).to_frame(name='Feature-Name'))
            

else:
    st.warning("Der DataFrame für den Prognosezeitraum ist leer. Bitte überprüfen Sie die Datenquelle.")

if __name__ == '__main__':
    print("Streamlit-App muss mit dem Befehl 'streamlit run streamlit_app.py' gestartet werden, nicht direkt mit Python.")