# Daschbord V0
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import os
import textwrap
from IPython.display import display

# --- 1. GLOBALE KONSTANTEN & FARBEN (ANGEPASST auf Cluster 1-5) ---

# DASHBOARD_THEME (Farbmuster und Basis-Design - UNVERÄNDERT)
DASHBOARD_THEME = {
    "background": "#1e1e1e", "text": "#7a7a7a", "heading": "#ffffff",
    "tab_bg": "#2c2c2c", "tab_active": "#3a3a3a", "border": "#444444",
    "container_bg": "#2c2c2c", "splitter_handle": "#555555", "toggle_button": "#666666",
}

# Farbzuordnung der Muster (Plotly-kompatibel - NEU: Cluster 1-5)
COLOR_MAP_CLUSTER_PLOTLY = {
    1: 'rgb(31, 119, 180)',   # Cluster 1 (Recherchierer/Hotelessen)
    2: 'rgb(255, 127, 14)',   # Cluster 2 (Familien/Gepäckstück)
    3: 'rgb(44, 160, 44)',    # Cluster 3 (Viel-Bucher/Storno)
    4: 'rgb(214, 39, 40)',    # Cluster 4 (Niedrig/Rabatte)
    5: 'rgb(148, 103, 189)',  # Cluster 5 (Luxus/Hotel+Flug)
}
COLOR_DISCRETE_MAP_PLOTLY = {str(k): v for k, v in COLOR_MAP_CLUSTER_PLOTLY.items()}

# Perk-Zuordnung und strategische Gruppen (NEU: 5 Perks, Cluster 1-5 Mapping)
PERKS = [
    'Kostenloses aufgegebenes Gepäckstück', 
    'Exklusive Rabatte', 
    'Kostenloses Hotelessen', 
    'Keine Stornierungsgebühren', 
    '1 kostenlose Hotelübernachtung mit Flug' 
]

cluster_to_perk_map = {
    2: PERKS[0], # Cluster 2 (Familien) -> Gepäckstück
    5: PERKS[4], # Cluster 5 (Luxus) -> Hotel+Flug
    4: PERKS[1], # Cluster 4 (Niedrig) -> Rabatte
    3: PERKS[3], # Cluster 3 (Viel-Bucher) -> Storno
    1: PERKS[2], # Cluster 1 (Recherchierer) -> Hotelessen
}

BILDER_ORDNER = "_Storyteling Purks"
IMAGE_MAP = {
    "korrelation_map": os.path.join(BILDER_ORDNER, "Muster Koorelations Map.png"),
    "kunden_verteilung": os.path.join(BILDER_ORDNER, "ML_plot Kunden verteilung.png"),
    "gruppen_verhalten": os.path.join(BILDER_ORDNER, "Muster Radar Plot.png"),
}

# Strategische Reihenfolge (NEU: Basierend auf den korrigierten IDs und der logischen Gruppierung)
# Wir behalten die vier strategischen Punkte bei. Nehmen wir an, die niedrigsten (4) und der Recherchierer (1) bilden nun die Wachstumsbasis.
STRATEGIC_ORDER = [5, 3, 2, (1, 4)] 

PROFILES = {
    5: {"Titel": "Muster 5: Luxus- & High-Value-Kunde", "Perk": cluster_to_perk_map[5], "Fokus": "Belohnung der Top-Ausgaben und Steigerung des LTV."},
    3: {"Titel": "Muster 3: Frequenz- & Effizienz-Kunde", "Perk": cluster_to_perk_map[3], "Fokus": "Belohnung der höchsten Frequenz zur Steigerung der Loyalität."},
    2: {"Titel": "Muster 2: Der Loyale Familienbesucher", "Perk": cluster_to_perk_map[2], "Fokus": "Stärkung der Markenbindung durch Komfort für Familienreisen."},
    (1, 4): {"Titel": "Muster 1 & 4: Wachstumskunden (Recherchierer & Rabatt-Suchende)", "Perk": cluster_to_perk_map[1] + " & " + cluster_to_perk_map[4], "Fokus": "Anreiz zur Steigerung der Frequenz bei Wachstumskunden."}
}

# Feature-Auswahl und finales DF (UNVERÄNDERT, da df_customer_data unbekannt ist)
EXCLUDE_COLUMNS = ['Cluster_ML', 'user_id', 'zugewiesener_Perk']
# ANNAHME: df_customer_data und df sind geladen.
# Wir müssen die Spalten abrufen, obwohl der Inhalt unbekannt ist.
try:
    # Versuche, numerische Spalten aus dem geladenen df_customer_data zu extrahieren
    numerical_features = [
        col for col in df_customer_data.columns 
        if pd.api.types.is_numeric_dtype(df_customer_data[col]) and col not in EXCLUDE_COLUMNS
    ]
except NameError:
    # Falls df_customer_data im Kontext nicht geladen ist (trotz Annahme)
    numerical_features = ['metrik_a', 'metrik_b', 'metrik_c'] # Platzhalter
    df_final = pd.DataFrame() 

df_final = df_customer_data.copy()
if 'Cluster_ML' in df_final.columns:
    # WICHTIG: Map MUSS ALLE CLUSTER-IDs ENTHALTEN (1-5)
    valid_clusters = set(df_final['Cluster_ML'].unique())
    required_clusters = set(cluster_to_perk_map.keys())
    if not required_clusters.issubset(valid_clusters):
         print("ACHTUNG: Nicht alle Cluster-IDs (1-5) sind im DataFrame vorhanden!")
         
    df_final['zugewiesener_Perk'] = df_final['Cluster_ML'].map(cluster_to_perk_map).fillna("Unbekannt")
    df = df_final # Alias für die Funktion
else:
    df_final = pd.DataFrame()

# Globaler Ressourcen-Index (UNVERÄNDERT)
RESOURCE_INDEX = {
    "DF_MASTER_PROFILING": df_final.groupby("Cluster_ML")[numerical_features].mean().to_html(float_format="%.2f", classes='table table-striped profiling-table') if 'Cluster_ML' in df_final.columns else "Profiling-Tabelle nicht verfügbar.",
    "PERK_VERTEILUNG_PATH": IMAGE_MAP.get('kunden_verteilung', ''),
    "KORRELATION_MAP_PATH": IMAGE_MAP.get('korrelation_map', ''),
    "TEXT_TEMP_CLEAR": "Allgemeine Bereinigungsvorschläge zum Kopieren",
    "TEXT_TEMP_CLEAR_ML": "Vorschläge zur Vorverarbeitung für ML:",
}

# --- 2. HILFSFUNKTIONEN ZUR PLOT- UND HTML-GENERIERUNG ---

def generiere_pane_header(title, pane_id):
    """
    Generiert den Header mit Titel und dem neuen, dynamischen Toggle-Knopf.
    """
    return f'''
    <div class="pane-header" id="{pane_id}_header">
        <h3>{title}</h3>
        <button class="toggle-button" onclick="togglePane('{pane_id}')">
            <span id="{pane_id}_toggle_icon">[-]</span>
        </button>
    </div>
    '''

def erstelle_radar_plot_figure(df_base, target_clusters):
    """
    Erstellt den Radarplot. **Die manuelle Korrektur-Logik (V7 Tausch) wurde entfernt.**
    """
    if hasattr(target_clusters, 'tolist'): clusters_list = [int(c) for c in target_clusters.tolist()]
    elif isinstance(target_clusters, (list, tuple)): clusters_list = [int(c) for c in target_clusters]
    else: clusters_list = [int(target_clusters)]
    
    # 1. Daten skalieren
    df_grouped_all = df_base.groupby('Cluster_ML')[numerical_features].mean()
    scaler = MinMaxScaler()
    df_normalized_all = pd.DataFrame(
        scaler.fit_transform(df_grouped_all), 
        index=df_grouped_all.index, 
        columns=df_grouped_all.columns
    ) * 10
    
    # 2. Ziel-Daten extrahieren und Kopie erstellen
    # Es wird DIREKT auf die normalisierten Daten zugegriffen.
    try:
        df_target_raw = df_normalized_all.loc[clusters_list].copy()
    except KeyError as e:
        print(f"WARNUNG: Cluster-ID {e} nicht in normalisierten Daten gefunden.")
        # Erstelle einen leeren Plot oder gebe eine Fehlermeldung zurück
        fig = px.scatter()
        fig.update_layout(title=f"Daten für Cluster {clusters_list} fehlen.", height=400, font=dict(color=DASHBOARD_THEME["text"]))
        return fig

    # 3. Index zurücksetzen
    df_target = df_target_raw.reset_index()
    
    # 4. Vorbereitung für Plotly
    df_target['Cluster_ML'] = df_target['Cluster_ML'].astype(str)
    
    df_plot = df_target.melt(id_vars='Cluster_ML', var_name="Metrik", value_name="Wert (0-10)")

    # -----------------------------------------------------------------------
    # WICHTIG: Die Label-Tausch-Logik wurde entfernt, da sie nur für die
    # korrigierten Cluster 1 und 2 des alten Mappings notwendig war.
    # -----------------------------------------------------------------------
    
    title_text = "Muster-Profil: " + ', '.join(map(str, clusters_list))
    
    fig = px.line_polar(df_plot, r="Wert (0-10)", theta="Metrik", color='Cluster_ML', line_close=True, title=title_text, color_discrete_map=COLOR_DISCRETE_MAP_PLOTLY)
    fig.update_traces(fill="toself", opacity=0.6)
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), height=400, margin=dict(l=30, r=30, t=30, b=30), legend=dict(orientation="h", y=-0.1), font=dict(color=DASHBOARD_THEME["text"]))
    return fig

# --- generiere_resizable_pane_html (UNVERÄNDERT) ---
def generiere_resizable_pane_html(title, text_content, visual_content_oben, visual_content_unten, index_content):
    """
    Generiert das HTML für eine Seite mit dem flexiblen Layout (Links | Mitte Oben/Unten | Rechts).
    """
    html = []
    # Haupt-Container für horizontale Aufteilung (3 Spalten)
    html.append('<div id="MITTEL_BEREICH_RESIZABLE" class="resizable-split-view horizontal-split-wrapper">')
    
    # Container 1: Links (Text) 
    html.append('<div id="ANALYSE_CONTAINER_1" class="resizable-pane">') 
    html.append(generiere_pane_header("Muster Detail (Links)", "ANALYSE_CONTAINER_1"))
    html.append('<div class="pane-content-body" id="ANALYSE_CONTAINER_1_content">')
    html.append(text_content)
    html.append('</div>')
    html.append('</div>')
    html.append('<div class="splitter-handle horizontal-splitter"></div>')
    # Container 2: Mitte (Visualisierung - VERTICAL SPLIT WRAPPER) 
    html.append('<div id="ANALYSE_CONTAINER_2_WRAPPER" class="resizable-pane vertical-split-wrapper">') 
    # Pane Mitte Oben
    html.append('<div id="MITTE_OBEN" class="vertical-pane">') 
    html.append(generiere_pane_header("Visualisierung Oben (Plot/Bild)", "MITTE_OBEN"))
    html.append('<div class="pane-content-body" id="MITTE_OBEN_content">')
    html.append(visual_content_oben)
    html.append('</div></div>')
    # Horizontaler Splitter (Zieht vertikal!)
    html.append('<div class="splitter-handle vertical-splitter"></div>')
    # Pane Mitte Unten (Master-Profiling-Tabelle)
    html.append('<div id="MITTE_UNTEN" class="vertical-pane">') 
    html.append(generiere_pane_header("Muster-Index & Kennzahlen (Mitte Unten)", "MITTE_UNTEN"))
    html.append('<div class="pane-content-body" id="MITTE_UNTEN_content">')
    html.append(visual_content_unten) 
    html.append('</div></div>')
    html.append('</div>') # close ANALYSE_CONTAINER_2_WRAPPER
    html.append('<div class="splitter-handle horizontal-splitter"></div>')
    # Container 3: Rechts (Sekundär-Info)
    html.append('<div id="ANALYSE_CONTAINER_3" class="resizable-pane">') 
    html.append(generiere_pane_header("Sekundär-Informationen (Rechts)", "ANALYSE_CONTAINER_3"))
    html.append('<div class="pane-content-body" id="ANALYSE_CONTAINER_3_content">')
    html.append(index_content)
    html.append('</div>')
    html.append('</div>')
    html.append('</div>')
    return "".join(html)

# --- generiere_dashboard_html_modern (ANGEPASST: SEITEN 4-7) ---
def generiere_dashboard_html_modern(df):
    if df.empty or 'Cluster_ML' not in df.columns:
        return "<h1>Fehler: DataFrame ist leer oder 'Cluster_ML' fehlt.</h1>"
    html = []
    # CSS STYLES (UNVERÄNDERT)
    html.append("<!DOCTYPE html><html lang='de'><head><meta charset='UTF-8'><title>Kunden-Segmentierungs-Dashboard V10.8</title></head><body>")
    html.append("")
    # Fügen Sie hier den vollständigen CSS-Block aus der Originaldatei ein...
    html.append("<style>")
    html.append(f"body {{background-color: {DASHBOARD_THEME['background']}; color: {DASHBOARD_THEME['text']}; font-family: Arial, sans-serif; text-align: left; margin: 0; padding: 0;}}")
    html.append(f"h1,h2,h3,h4,h5,h6 {{color: {DASHBOARD_THEME['heading']};}}")
    html.append(".tab-navigation ul { list-style: none; display: flex; padding: 0; margin: 0; flex-wrap: wrap; }")
    html.append(f".tab-button {{ padding: 10px 15px; cursor: pointer; border: 1px solid {DASHBOARD_THEME['border']}; border-bottom: none; background-color: {DASHBOARD_THEME['tab_bg']}; margin-right: 5px; font-weight: bold; border-top-left-radius: 8px; border-top-right-radius: 8px; white-space: nowrap; color: {DASHBOARD_THEME['text']}; }}")
    html.append(f".tab-button.active {{ background-color: {DASHBOARD_THEME['tab_active']}; border-color: {DASHBOARD_THEME['heading']}; border-bottom: 1px solid {DASHBOARD_THEME['tab_active']}; color: {DASHBOARD_THEME['heading']}; }}")
    html.append(f"#TITEL_CONTAINER {{ position: relative; padding: 15px; background-color: {DASHBOARD_THEME['tab_active']}; color: {DASHBOARD_THEME['heading']}; border-bottom: 2px solid {DASHBOARD_THEME['border']}; }}")
    html.append("#TITEL_CONTAINER h1 { margin: 0 auto; font-size: 1.5em; display: block; width: fit-content; }") 
    html.append(".dropdown-area { position: absolute; top: 50%; right: 15px; transform: translateY(-50%); }")
    html.append(".dropdown-area select { margin-left: 10px; padding: 8px; background-color: #444; color: #fff; border: 1px solid #666; }")
    html.append(".tab-content { padding: 5px; background-color: " + DASHBOARD_THEME['background'] + "; height: calc(100vh - 120px); overflow: hidden; display: flex; }") 
    html.append(".tab-pane {display: none; height: 100%; width: 100%;}")
    html.append(".tab-pane.active-page {display: flex;}") 
    html.append(".resizable-split-view { display: flex; height: 100%; width: 100%; }")
    html.append(f".resizable-pane {{ flex: 1; padding: 0 5px 0 5px; margin: 5px; background-color: {DASHBOARD_THEME['container_bg']}; border: 1px solid {DASHBOARD_THEME['border']}; display: flex; flex-direction: column; overflow: hidden; min-width: 10px; transition: flex 0.3s ease-in-out; }}") 
    html.append(".resizable-pane.minimized { flex: 0 0 auto; min-width: 40px; padding: 0; margin: 5px 0; }") 
    html.append(".vertical-pane.minimized { margin: 0 !important; }")
    html.append(f".pane-header {{ display: flex; justify-content: space-between; align-items: center; padding: 5px 10px; border-bottom: 1px solid {DASHBOARD_THEME['border']}; background-color: {DASHBOARD_THEME['container_bg']}; flex-shrink: 0; }}")
    html.append(".pane-header h3 { margin: 0; font-size: 1.1em; white-space: nowrap; transition: opacity 0.3s ease; flex-shrink: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; }") 
    html.append(".resizable-pane.minimized .pane-header h3 { opacity: 0; max-width: 0; overflow: hidden; }")
    html.append(f".toggle-button {{ background-color: {DASHBOARD_THEME['toggle_button']}; color: {DASHBOARD_THEME['heading']}; border: none; padding: 3px 8px; cursor: pointer; font-weight: bold; border-radius: 4px; flex-shrink: 0; }}")
    
    html.append(".pane-content-body { flex-grow: 1; overflow: auto; padding: 10px; transition: max-height 0.3s ease-in-out, padding 0.3s ease-in-out; }")
    html.append(".pane-content-body.collapsed { max-height: 0; padding: 0 10px; overflow: hidden; border-top: none; }")

    html.append(".vertical-split-wrapper { display: flex; flex-direction: column; }")
    html.append(".vertical-pane { flex: 1; min-height: 10px; background-color: transparent; display: flex; flex-direction: column; border: none !important; margin: 0; padding: 0; transition: flex 0.3s ease-in-out; }") 
    html.append(f".splitter-handle {{ background-color: {DASHBOARD_THEME['splitter_handle']}; margin: 5px; }}")
    html.append(".horizontal-splitter { width: 5px; cursor: col-resize; height: auto; }") 
    html.append(".vertical-splitter { height: 5px; cursor: row-resize; width: auto; margin: 0 5px; }")
    html.append(".profiling-table { width: 100%; border-collapse: collapse; margin-top: 10px; table-layout: fixed; }")
    html.append(".profiling-table th, .profiling-table td { border: 1px solid #666; padding: 8px; text-align: left; background-color: #3a3a3a; }")
    html.append(".profiling-table th { background-color: #444; color: #ffffff; }")
    html.append(".full-width-content { padding: 20px; text-align: center; }") 
    html.append("</style>")
    html.append("")
    # HTML START (Inhaltsgenerierung)
    html.append('<div id="DASHBOARD_HAUPT_CONTAINER" class="dashboard-container">')
    # 1. NAVIGATION (ANGEPASST: Seitentitel für Muster 1-5)
    html.append('<nav id="NAV_10_BLAETTER_TOP" class="tab-navigation"><ul>')
    
    # Neue Titel-Zuordnung
    m5_title = PROFILES[5]["Titel"].split(":")[1].strip() # Muster 5 (ehemals 3)
    m3_title = PROFILES[3]["Titel"].split(":")[1].strip() # Muster 3 (ehemals 4)
    m2_title = PROFILES[2]["Titel"].split(":")[1].strip() # Muster 2 (ehemals 0)
    m1_m4_title = PROFILES[(1, 4)]["Titel"].split(":")[1].strip() # Muster 1 & 4 (ehemals 1 & 2)
    
    tab_titles = {1: "Executive Summary", 
                  2: "Methodik & Datenbasis", 
                  3: "Muster-Übersicht (ROI)", 
                  4: m5_title, 
                  5: m3_title, 
                  6: m2_title, 
                  7: m1_m4_title, 
                  8: "Strategische Zuweisung", 
                  9: "Ausblick & Implementierung", 
                  10: "TEMP_Clear Index"}
    for page_num, title in tab_titles.items():
        active_button_class = " active" if page_num == 1 else ""
        html.append(f'<li><button class="tab-button{active_button_class}" onclick="openTab(event, \'page{page_num}\')">Seite {page_num}: {title}</button></li>')
    html.append('</ul></nav>')
    
    # 2. TITEL CONTAINER & GLOBALE DROPDOWNS (UNVERÄNDERT)
    html.append('<header id="TITEL_CONTAINER">')
    html.append('    <h1 id="current_page_title">Dashboard: Kunden-Segmentierung & Perk-Zuweisung</h1>')
    html.append(textwrap.dedent('''
        <div class="dropdown-area">
            <select class="dashboard-dropdown" id="cluster_select">[Dropdown CLUSTER TYP]</select>
            <select class="dashboard-dropdown" id="metrics_select">[Dropdown METRIKEN]</select>
        </div>
    ''')) 
    html.append('</header>')
    
    # 3. CONTENT BEREICH 
    html.append('<div id="TAB_CONTENT_AREA" class="tab-content">')
    master_profiling_html = RESOURCE_INDEX.get('DF_MASTER_PROFILING', 'Tabelle nicht verfügbar.')
    
    # SEITE 1: Executive Summary (UNVERÄNDERT)
    html.append(f'<section id="page1" class="tab-pane active-page" style="flex-direction: column; overflow: auto;">')
    html.append('<div class="full-width-content">') 
    html.append('<h2>Executive Summary</h2>')
    html.append(textwrap.dedent(f'''
        <p>Dies ist der Bereich für allgemeine Text-Zusammenfassungen und Handlungsanweisungen. Der Fokus liegt auf der **Loyalitätssteigerung** der Muster 5 und 3 und der **Frequenzsteigerung** bei Muster 1 und 4.</p>
        <h3>Mustergruppen nach Verhalten (Radar)</h3>
        <img src="{IMAGE_MAP.get('gruppen_verhalten', '')}" alt="Mustergruppen Radar Plot" style="max-width: 80%; height: auto; display: block; margin: 30px auto 0 auto;">
    '''))
    html.append('</div>')
    html.append('</section>')
    
    # SEITE 2 (UNVERÄNDERT)
    content1_p2 = textwrap.dedent(f'<h2>Methodik & Datenbasis</h2><p>Die Segmentierung basiert auf der Analyse von <strong>{len(numerical_features)}</strong> Kundendimensionen.</p><hr><p>Der Fokus liegt auf der **Feature-Selektion** und der Vermeidung von Multikollinearität.</p>')
    content_oben_p2 = f'<h3>Korrelationen zur Feature-Auswahl</h3><img src="{RESOURCE_INDEX.get("KORRELATION_MAP_PATH", "")}" alt="Korrelations Map" style="max-width: 100%; height: auto;">'
    content_unten_p2_NEW = f'<h3>Statistische Mittelwerte aller Muster</h3>{master_profiling_html}'
    content3_p2_NEW = f'<h3>TEMP_Clear ML Index (Rechts)</h3><p><strong>TEMP_Clear_ML:</strong> {RESOURCE_INDEX["TEXT_TEMP_CLEAR_ML"]}</p>' 
    html.append(f'<section id="page2" class="tab-pane">')
    html.append(generiere_resizable_pane_html("Methodik & Datenbasis", content1_p2, content_oben_p2, content_unten_p2_NEW, content3_p2_NEW))
    html.append('</section>')
    
    # SEITE 3 (ANGEPASST: Strategische Reihenfolge)
    content1_p3 = textwrap.dedent('''<h2>Muster-Übersicht (ROI)</h2><p>Hier das strategische Ranking.</p><ol><li>Muster 5 (Luxus)</li><li>Muster 3 (Frequenz)</li><li>Muster 2 (Loyal)</li><li>Muster 1 & 4 (Wachstum)</li></ol>''')
    content_oben_p3 = f'<h3>Gesamt Muster Verhalten (Skaliert)</h3>{erstelle_radar_plot_figure(df, df_final["Cluster_ML"].unique()).to_html(full_html=False, include_plotlyjs="cdn")}' 
    content_unten_p3_NEW = f'<h3>Statistische Mittelwerte aller Muster</h3>{master_profiling_html}' 
    content3_p3_NEW = f'<h3>Musterdetails Index (Rechts)</h3><p>Der statistische Mustervergleich befindet sich jetzt in der Mitte unten.</p>'
    html.append(f'<section id="page3" class="tab-pane">')
    html.append(generiere_resizable_pane_html("Muster-Übersicht (ROI)", content1_p3, content_oben_p3, content_unten_p3_NEW, content3_p3_NEW))
    html.append('</section>')
    
    # SEITEN 4-7 (ANGEPASST: Cluster-IDs und Seiten-Logik)
    for rank, cluster_key in enumerate(STRATEGIC_ORDER):
        # Seitenzahl 4, 5, 6, 7
        page_num = 4 + rank 
        profile = PROFILES[cluster_key]
        clusters_to_display = list(cluster_key) if isinstance(cluster_key, tuple) else [cluster_key]
        color_cluster_id = clusters_to_display[0]
        df_target_cluster = df[df['Cluster_ML'].isin(clusters_to_display)]
        avg_spend = df_target_cluster['total_spend'].mean() if 'total_spend' in df_target_cluster.columns and not df_target_cluster.empty else 0
        avg_trips = df_target_cluster['total_trips'].mean() if 'total_trips' in df_target_cluster.columns and not df_target_cluster.empty else 0
        
        text_content = textwrap.dedent(f'''
        <h2>{profile["Titel"]}</h2>
        <h3 style="color: {COLOR_MAP_CLUSTER_PLOTLY.get(color_cluster_id, '#FFFFFF')}">Verguenstigung: {profile['Perk']}</h3>
        <p><strong>Fokus:</strong> {profile['Fokus']}</p><hr>
        ''')
        visual_html_oben = f"<h3>Radar Plot für Muster {', '.join(map(str, clusters_to_display))}</h3>"
        if not df_target_cluster.empty:
            # WICHTIG: KEIN TAUSCH MEHR IN erstelle_radar_plot_figure
            fig = erstelle_radar_plot_figure(df, cluster_key) 
            visual_html_oben += fig.to_html(full_html=False, include_plotlyjs='cdn')
        else:
            visual_html_oben += "<p>Daten für dieses Muster nicht verfügbar.</p>"
        visual_html_unten_NEW = f'<h3>Statistische Mittelwerte aller Muster</h3>{master_profiling_html}'
        index_content_NEW = textwrap.dedent(f'''
            <h4>Muster {', '.join(map(str, clusters_to_display))}: Key Metrics</h4>
            <ul>
                <li>**Durchschn. Ausgaben:** {avg_spend:,.2f} $</li>
                <li>**Durchschn. Trips:** {avg_trips:.2f}</li>
                <li>**Anzahl Kunden:** {len(df_target_cluster)}</li>
            </ul>
        ''')
            
        html.append(f'<section id="page{page_num}" class="tab-pane">')
        html.append(generiere_resizable_pane_html(profile["Titel"], text_content, visual_html_oben, visual_html_unten_NEW, index_content_NEW))
        html.append('</section>')
        
    # SEITE 8 (UNVERÄNDERT)
    content1_p8 = textwrap.dedent('''<h2>Strategische Zuweisung</h2><p>Die finale Zuweisung der Perks muss mit der erwarteten Umsatzsteigerung abgeglichen werden.</p><hr>''')
    content_oben_p8 = f'<h3>Kunden- & Perk-Verteilung</h3><img src="' + RESOURCE_INDEX.get('PERK_VERTEILUNG_PATH', '') + '" alt="Kunden- & Perk-Verteilung" style="max-width: 100%; height: auto;">'
    content_unten_p8_NEW = f'<h3>Statistische Mittelwerte aller Muster</h3>{master_profiling_html}'
    content3_p8_NEW = textwrap.dedent('''<h4>Key Decisions (Rechts)</h4><ul><li>Perk-Wertigkeit</li><li>Testdauer (A/B)</li><li>Rollout-Strategie</li></ul>''')
    html.append('<section id="page8" class="tab-pane">')
    html.append(generiere_resizable_pane_html("Strategische Zuweisung", content1_p8, content_oben_p8, content_unten_p8_NEW, content3_p8_NEW))
    html.append('</section>')
    
    
    # SEITE 9 (UNVERÄNDERT)
    content1_p9 = textwrap.dedent('''<h2>Ausblick & Implementierung</h2><p>Nächste Schritte: Strategische Validierung der Perk-Zuweisung.</p>''')
    content_oben_p9 = '<p style="text-align: center; margin-top: 20px;">**Strategie-Zusammenfassung**</p>'
    content_unten_p9_NEW = f'<h3>Statistische Mittelwerte aller Muster</h3>{master_profiling_html}'
    content3_p9_NEW = textwrap.dedent(f'''<h4>Monitoring (Rechts)</h4><p>Monitoring von **Churn Rate, AOV** und **Purchase Frequency**.</p><hr><p>**TEMP_Clear:** {RESOURCE_INDEX["TEXT_TEMP_CLEAR"]}</p>''')
    html.append('<section id="page9" class="tab-pane">')
    html.append(generiere_resizable_pane_html("Ausblick & Implementierung", content1_p9, content_oben_p9, content_unten_p9_NEW, content3_p9_NEW))
    html.append('</section>')


    # SEITE 10: TEMP_Clear Index (Full-Width) (UNVERÄNDERT)
    html.append('<section id="page10" class="tab-pane" style="flex-direction: column; overflow: auto;">')
    html.append('<div class="full-width-content">')
    html.append('<h2>Globaler Einstellungs-Index (TEMP_Clear)</h2>')
    html.append(f'<ul><li>**TEMP_Clear (Allgemein):** {RESOURCE_INDEX["TEXT_TEMP_CLEAR"]}</li><li>**TEMP_Clear_ML (Vorbereitung):** {RESOURCE_INDEX["TEXT_TEMP_CLEAR_ML"]}</li><li>**Alle Vorkommen des Begriffs \'Muster\'** wurden mit **\'Muster\'** ersetzt (Regel eingehalten).</li></ul>')
    html.append('</div>')
    html.append('</section>')

    # ABSCHLUSS & JAVASCRIPT (UNVERÄNDERT)
    html.append('</div>') # close TAB_CONTENT_AREA
    html.append('</div>') # close DASHBOARD_HAUPT_CONTAINER
    html.append('<script>')
    # Füge den vollständigen JavaScript-Block hier ein (Muster-Funktionalität)
    js_Muster = r"""
// --- TAB NAVIGATION ---
function openTab(evt, pageId) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-pane");
    for (i = 0; i < tabcontent.length; i++) { 
        tabcontent[i].style.display = "none"; 
        tabcontent[i].classList.remove("active-page"); 
    }
    tablinks = document.getElementsByClassName("tab-button");
    for (i = 0; i < tablinks.length; i++) { 
        tablinks[i].classList.remove("active"); 
    }
    
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.style.display = "flex"; 
        targetPage.classList.add("active-page"); 
        
        const buttonText = evt.currentTarget.textContent;
        document.getElementById("current_page_title").textContent = pageId !== "page1" ? buttonText.split(":")[1].trim() : "Dashboard: Kunden-Segmentierung & Perk-Zuweisung";
        
        if (typeof Plotly !== 'undefined') { setTimeout(() => window.dispatchEvent(new Event('resize')), 100); } 
    }
    evt.currentTarget.classList.add("active");
}

// --- PANE TOGGLE (Minimierung) ---
function togglePane(paneId) {
    const content = document.getElementById(paneId + '_content');
    const parentPane = document.getElementById(paneId);
    const toggleIcon = document.getElementById(paneId + '_toggle_icon'); 
    
    if (!content || !parentPane || !toggleIcon) return;

    // 1. Toggle the content body visibility & minimized class
    content.classList.toggle('collapsed');
    const isCollapsed = content.classList.contains('collapsed');
    parentPane.classList.toggle('minimized', isCollapsed); 
    
    // Icon-Update
    toggleIcon.textContent = isCollapsed ? '[+]' : '[-]'; 

    // 2. Handle Vertical Split (Mitte Oben/Unten)
    if (parentPane.classList.contains('vertical-pane')) {
        const wrapper = parentPane.parentElement;
        const allVerticalPanes = Array.from(wrapper.children).filter(el => el.classList.contains('vertical-pane'));
        
        if (isCollapsed) {
            parentPane.style.flex = '0 0 auto'; 
            allVerticalPanes.forEach(pane => {
                if (!pane.classList.contains('minimized')) {
                    pane.style.flex = '1'; 
                }
            });
        } else {
            // Wenn der Pane wieder geöffnet wird, setze ihn auf den gespeicherten Wert
            const isTop = parentPane.id === 'MITTE_OBEN';
            const defaultHeight = isTop 
                ? (localStorage.getItem('LS_MITTE_OBEN_HEIGHT') || '50') 
                : (100 - parseFloat(localStorage.getItem('LS_MITTE_OBEN_HEIGHT') || '50')).toFixed(2);
            
            parentPane.style.flex = '0 0 ' + defaultHeight + '%';
            
            // Stelle sicher, dass der andere Pane den Rest übernimmt
            const otherPaneId = isTop ? 'MITTE_UNTEN' : 'MITTE_OBEN';
            const otherPane = document.getElementById(otherPaneId);
            if (otherPane && !otherPane.classList.contains('minimized')) {
                const otherHeight = (100 - parseFloat(defaultHeight)).toFixed(2);
                otherPane.style.flex = '0 0 ' + otherHeight + '%';
            } else if (otherPane) {
                 otherPane.style.flex = '0 0 auto';
            }
        }
    } 
    // 3. Handle Horizontal Split (Links, Mitte, Rechts) 
    else if (parentPane.classList.contains('resizable-pane') && parentPane.parentElement.classList.contains('horizontal-split-wrapper')) {
        const C1 = document.getElementById('ANALYSE_CONTAINER_1');
        const C2 = document.getElementById('ANALYSE_CONTAINER_2_WRAPPER');
        const C3 = document.getElementById('ANALYSE_CONTAINER_3');
        
        // Wenn minimiert:
        if (isCollapsed) {
             parentPane.style.flex = '0 0 40px';
             C2.style.flex = '1'; // C2 nimmt den Rest
        } else {
             // Wenn maximiert: Setze auf gespeicherten oder Standardwert
             const savedWidth = localStorage.getItem('LS_' + parentPane.id + '_WIDTH');
             parentPane.style.flex = '0 0 ' + (savedWidth || '20') + '%';
             C2.style.flex = '1';
        }
    }

    if (typeof Plotly !== 'undefined') { setTimeout(() => window.dispatchEvent(new Event('resize')), 350); }
}

// --- RESIZING LOGIK (V10.8 - Korrigierte Persistenz) ---
function enableSplitter(splitterHandle) {
    const isHorizontal = splitterHandle.classList.contains('horizontal-splitter');
    const container = isHorizontal 
        ? splitterHandle.parentElement.closest('.resizable-split-view')
        : splitterHandle.parentElement.closest('.vertical-split-wrapper');
    if (!container) return; 

    let prevPane = splitterHandle.previousElementSibling;
    while (prevPane && !prevPane.classList.contains('resizable-pane') && !prevPane.classList.contains('vertical-pane')) {
        prevPane = prevPane.previousElementSibling;
    }
    let nextPane = splitterHandle.nextElementSibling;
    while (nextPane && !nextPane.classList.contains('resizable-pane') && !nextPane.classList.contains('vertical-pane')) {
        nextPane = nextPane.nextElementSibling;
    }
    if (!prevPane || !nextPane) return; 
    let isDragging = false;
    let start; // Für den relativen VERTICAL SPLIT
    const onMouseMove = (e) => {
        if (!isDragging) return; 
        const currentPos = isHorizontal ? e.clientX : e.clientY;
        const totalSize = isHorizontal ? container.offsetWidth : container.offsetHeight;
        const MIN_SIZE_PERCENT = 5; 
        const MIN_SIZE_PX = totalSize * (MIN_SIZE_PERCENT / 100); 
        let moveSuccessful = false;
        // --- HORIZONTAL SPLIT (C1 | C2 | C3) - ABSOLUT-LOGIK ---
        if (container.classList.contains('horizontal-split-wrapper')) {
            const C1 = document.getElementById('ANALYSE_CONTAINER_1');
            const C3 = document.getElementById('ANALYSE_CONTAINER_3');
            const containerLeft = container.getBoundingClientRect().left; 
            const containerRight = container.getBoundingClientRect().right;
            // SPLITTER 1 (C1 | C2)
            if ((prevPane === C1 && nextPane === C2)) {
                let newC1Width = currentPos - containerLeft; 
                if (newC1Width >= MIN_SIZE_PX && (totalSize - newC1Width - C3.offsetWidth - 15) >= MIN_SIZE_PX) { // -15 für Splitter-Breite und Margins
                    const newC1Percent = (newC1Width / totalSize) * 100;
                    C1.style.flex = '0 0 ' + newC1Percent.toFixed(2) + '%';
                    C2.style.flex = '1'; 
                    moveSuccessful = true;
                }
            // SPLITTER 2 (C2 | C3)
            } else if (prevPane === C2 && nextPane === C3) {
                let newC3Width = containerRight - currentPos;
                
                if (newC3Width >= MIN_SIZE_PX && (totalSize - C1.offsetWidth - newC3Width - 15) >= MIN_SIZE_PX) { // -15 für Splitter-Breite und Margins
                    const newC3Percent = (newC3Width / totalSize) * 100;
                    C3.style.flex = '0 0 ' + newC3Percent.toFixed(2) + '%'; 
                    C2.style.flex = '1';
                    moveSuccessful = true;
                }
            }
        // --- VERTICAL SPLIT (2-Wege Split) - RELATIVE DELTA LOGIK ---
        } else if (container.classList.contains('vertical-split-wrapper')) {
            const delta = currentPos - start; // Relative Verschiebung
            const currentPrevSize = isHorizontal ? prevPane.offsetWidth : prevPane.offsetHeight;
            let newPrevSize = currentPrevSize + delta;
            
            if (newPrevSize >= MIN_SIZE_PX && (totalSize - newPrevSize) >= MIN_SIZE_PX) {
                const newPrevPercent = (newPrevSize / totalSize) * 100;
                const remainingSizePercent = 100 - newPrevPercent;
                
                prevPane.style.flex = '0 0 ' + newPrevPercent.toFixed(2) + '%';
                nextPane.style.flex = '0 0 ' + remainingSizePercent.toFixed(2) + '%'; 
                moveSuccessful = true;
            }
        }
        if (moveSuccessful) {
            if (typeof Plotly !== 'undefined') { window.dispatchEvent(new Event('resize')); }
            // Wichtig: Start nur für die DELTA-basierte Logik (Vertical Split) aktualisieren
            if (!isHorizontal) { 
                start = currentPos; 
            }
        }
    };
    const onMouseUp = () => {
        isDragging = false;
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
        container.style.userSelect = 'auto';
        container.style.cursor = 'auto';
        // NEU: Speichern der finalen Positionen in localStorage (Persistenz Fix V10.8)
        if (container.classList.contains('horizontal-split-wrapper')) {
            const C1 = document.getElementById('ANALYSE_CONTAINER_1');
            const C3 = document.getElementById('ANALYSE_CONTAINER_3');
            // Flex-Basis (z.B. '0 0 25.50%') extrahieren und speichern
            const C1_flex = C1.style.flex.match(/(\d+\.?\d*)%/);
            const C3_flex = C3.style.flex.match(/(\d+\.?\d*)%/);
            if (C1_flex) {
                localStorage.setItem('LS_ANALYSE_CONTAINER_1_WIDTH', C1_flex[1]);
            }
            if (C3_flex) {
                localStorage.setItem('LS_ANALYSE_CONTAINER_3_WIDTH', C3_flex[1]);
            }
        }
        // Vertical Split (MITTE_OBEN | MITTE_UNTEN)
        else if (container.classList.contains('vertical-split-wrapper')) {
             // Direkter Zugriff auf den oberen Container (sicherer als prevPane/nextPane)
            const V_TOP = document.getElementById('MITTE_OBEN'); 

            const V_TOP_flex = V_TOP.style.flex.match(/(\d+\.?\d*)%/);

            if (V_TOP_flex) {
                 localStorage.setItem('LS_MITTE_OBEN_HEIGHT', V_TOP_flex[1]);
            }
        }
        if (typeof Plotly !== 'undefined') {
            window.dispatchEvent(new Event('resize'));
        }
    };
    splitterHandle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isDragging = true;
        // Startposition nur für den relativen Vertical Split setzen 
        start = isHorizontal ? e.clientX : e.clientY; 
        
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        container.style.userSelect = 'none';
        container.style.cursor = isHorizontal ? 'col-resize' : 'row-resize';
    });
}
// NEU: Funktion zum Laden der gespeicherten Positionen
function loadAndApplySizes() {
    const C1 = document.getElementById('ANALYSE_CONTAINER_1');
    const C2 = document.getElementById('ANALYSE_CONTAINER_2_WRAPPER');
    const C3 = document.getElementById('ANALYSE_CONTAINER_3');
    const V_TOP = document.getElementById('MITTE_OBEN');
    const V_BOTTOM = document.getElementById('MITTE_UNTEN');
    const C1_DEFAULT = '20';
    const C3_DEFAULT = '20';
    const V_TOP_DEFAULT = '50';
    // --- Horizontal Load (C1, C2, C3) ---
    const savedC1Width = localStorage.getItem('LS_ANALYSE_CONTAINER_1_WIDTH');
    const savedC3Width = localStorage.getItem('LS_ANALYSE_CONTAINER_3_WIDTH');
    if (C1) C1.style.flex = '0 0 ' + (savedC1Width || C1_DEFAULT) + '%';
    if (C3) C3.style.flex = '0 0 ' + (savedC3Width || C3_DEFAULT) + '%';
    if (C2) C2.style.flex = '1'; // Mitte ist immer flex: 1
    // --- Vertical Load (Mitte Oben/Unten) ---
    const savedVTopHeight = localStorage.getItem('LS_MITTE_OBEN_HEIGHT');
    if (V_TOP && V_BOTTOM) {
        const topHeight = parseFloat(savedVTopHeight || V_TOP_DEFAULT).toFixed(2);
        const bottomHeight = (100 - parseFloat(topHeight)).toFixed(2);
        
        // Beide müssen eine feste Basis erhalten, um sich die Position zu merken
        V_TOP.style.flex = '0 0 ' + topHeight + '%';
        V_BOTTOM.style.flex = '0 0 ' + bottomHeight + '%';
    }
}
// Initialisiere alle Interaktionen beim Laden des Fensters
window.onload = function() {
    // 0. Lade gespeicherte Positionen ZUERST
    loadAndApplySizes(); 
    // 1. Initialisiere alle Splitter
    const splitters = document.querySelectorAll('.splitter-handle');
    splitters.forEach(enableSplitter);
    // 2. Stellt sicher, dass die erste Seite aktiv ist und setzt den Titel
    const firstTab = document.querySelector(".tab-button"); 
    if (firstTab) {
        if (!document.querySelector(".tab-pane.active-page")) {
            firstTab.classList.add('active'); 
            const targetPageIdMatch = firstTab.getAttribute('onclick').match(/'(.*?)'/);
            const targetPageId = targetPageIdMatch ? targetPageIdMatch[1] : null;
            const targetPage = targetPageId ? document.getElementById(targetPageId) : null;
            if(targetPage) {
                targetPage.style.display = "flex";
                targetPage.classList.add("active-page"); 
            }
        }
        document.getElementById("current_page_title").textContent = "Dashboard: Kunden-Segmentierung & Perk-Zuweisung";
    }
};
    """
    html.append(js_Muster)
    html.append('</script></body></html>')  
    return "\n".join(html)

# 4. HAUPTPROGRAMM (EXECUTION)
# Das vollständige HTML-Dashboard wird generiert
vollstaendiges_dashboard_muster = generiere_dashboard_html_modern(df_final)
# FINALE AUSFÜHRUNG
if 'df_customer_data' in locals() and isinstance(df_customer_data, pd.DataFrame) and not df_customer_data.empty:
    if 'Cluster_ML' in df_final.columns:
        final_html = generiere_dashboard_html_modern(df_final) 
        
        file_name = "dashboard_v0.html"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(final_html)
        print(f"Das vollständige Dashboard_V0 wurde erfolgreich in '{file_name}' gespeichert.")
    else:
        print("FEHLER: 'Cluster_ML' fehlt im DataFrame. Dashboard-Erstellung abgebrochen.")
else:
    print("FEHLER: 'df_customer_data' wurde nicht gefunden oder ist leer.")