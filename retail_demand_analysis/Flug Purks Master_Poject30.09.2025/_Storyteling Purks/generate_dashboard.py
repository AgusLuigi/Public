# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import os

# This script assumes 'df_customer_data' is a pandas DataFrame existing in the global scope.
if 'df_customer_data' not in locals():
    raise NameError("Das DataFrame 'df_customer_data' wurde nicht im globalen Scope gefunden.")

DASHBOARD_THEME = {
    "background": "#2b2929", "text": "#867070", "heading": "#F5F5F5",
    "tab_bg": "#f1f1f1", "tab_active": "#dddddd", "border": "#999999",
}

COLOR_MAP_CLUSTER_PLOTLY = {
    0: 'rgb(31, 119, 180)', 1: 'rgb(255, 127, 14)', 2: 'rgb(44, 160, 44)',
    3: 'rgb(214, 39, 40)', 4: 'rgb(148, 103, 189)',
}
COLOR_DISCRETE_MAP_PLOTLY = {str(k): v for k, v in COLOR_MAP_CLUSTER_PLOTLY.items()}

PERKS = [
    'Kostenloses aufgegebenes Gepäckstück', 'Exklusive Rabatte', 'Kostenloses Hotelessen',
    'Keine Stornierungsgebühren', '1 kostenlose Hotelübernachtung mit Flug'
]
cluster_to_perk_map = {i: PERKS[i] for i in range(len(PERKS))}

BILDER_ORDNER = "_Storyteling Purks"
IMAGE_MAP = {
    "korrelation_map": os.path.join(BILDER_ORDNER, "Muster Koorelations Map.png"),
    "kunden_verteilung": os.path.join(BILDER_ORDNER, "ML_plot Kunden verteilung.png"),
    "gruppen_verhalten": os.path.join(BILDER_ORDNER, "Muster Radar Plot.png"),
}

STRATEGIC_ORDER = [3, 4, 0, (1, 2)]
perk_1_2 = cluster_to_perk_map[1] + " (Muster 1) & " + cluster_to_perk_map[2] + " (Muster 2)"
PROFILES = {
    3: {"Titel": "Muster 3: Luxus- & Flexibilitäts-Kunde", "Perk": cluster_to_perk_map[3], "Fokus": "Belohnung der Top-Ausgaben und Absicherung des Flexibilitätsbedürfnisses."},
    4: {"Titel": "Muster 4: Reise-Effizienz & Frequenz", "Perk": cluster_to_perk_map[4], "Fokus": "Belohnung der höchsten Frequenz und Conversion Rate zur Steigerung der Loyalität."},
    0: {"Titel": "Muster 0: Der Engagierte Loyale", "Perk": cluster_to_perk_map[0], "Fokus": "Belohnung des Engagements und Stärkung der Markenbindung durch Komfort."},
    (1, 2): {"Titel": "Muster 1 & 2: Gelegentliche Bucher", "Perk": perk_1_2, "Fokus": "Anreiz zur Steigerung der Frequenz bei Wachstumskunden."
}

numerical_features = ["total_trips", "total_spend", "age", "session_duration_seconds", "page_clicks", "total_cancellations"]

df_final = df_customer_data.copy()
if 'Cluster_M' in df_final.columns:
    df_final['zugewiesener_Perk'] = df_final['Cluster_M'].map(cluster_to_perk_map)
else:
    print("FEHLER: Spalte 'Cluster_M' nicht in df_customer_data gefunden!")
    df_final = pd.DataFrame()

def erstelle_radar_plot_figure(df_base, target_clusters):
    df_grouped = df_base.groupby('Cluster_M')[numerical_features].mean()
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_grouped), index=df_grouped.index, columns=df_grouped.columns) * 10
    clusters_list = list(target_clusters) if isinstance(target_clusters, tuple) else [target_clusters]
    df_target = df_normalized.loc[clusters_list].reset_index()
    df_target['Cluster_M'] = df_target['Cluster_M'].astype(str)
    df_plot = df_target.melt(id_vars='Cluster_M', var_name="Metrik", value_name="Wert (0-10)")
    fig = px.line_polar(df_plot, r="Wert (0-10)", theta="Metrik", color='Cluster_M', line_close=True, color_discrete_map=COLOR_DISCRETE_MAP_PLOTLY, title="Muster-Profil: " + ', '.join(map(str, clusters_list)))
    fig.update_traces(fill="toself")
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), height=400, margin=dict(l=50, r=50, t=50, b=50), legend=dict(orientation="h", y=-0.1), font=dict(color=DASHBOARD_THEME["text"]))
    return fig

def generiere_dashboard_html(df):
    if df.empty or 'Cluster_M' not in df.columns:
        return "<h1>Fehler: DataFrame ist leer oder 'Cluster_M' fehlt.</h1>"
    
    cluster_profiling = df.groupby("Cluster_M")[numerical_features].mean()
    profiling_html = cluster_profiling.to_html(float_format="%.2f", classes='table table-striped')
    
    html = []
    html.append("<style>")
    html.append("body {background-color: " + DASHBOARD_THEME['background'] + "; color: " + DASHBOARD_THEME['text'] + "; font-family: Arial, sans-serif;}")
    html.append("h1,h2,h3,h4,h5,h6 {color: " + DASHBOARD_THEME['heading'] + ";}")
    html.append(".tab-buttons {display: flex; flex-wrap: wrap; margin-bottom: 20px; border-bottom: 2px solid " + DASHBOARD_THEME['border'] + ";}")
    html.append(".tab-button { padding: 10px 15px; cursor: pointer; border: 1px solid " + DASHBOARD_THEME['border'] + "; border-bottom: none; background-color: " + DASHBOARD_THEME['tab_bg'] + "; margin-right: 5px; font-weight: bold; border-top-left-radius: 8px; border-top-right-radius: 8px; white-space: nowrap; }")
    html.append(".tab-button.active { background-color: " + DASHBOARD_THEME['tab_active'] + "; border-color: " + DASHBOARD_THEME['heading'] + "; border-bottom: 1px solid " + DASHBOARD_THEME['background'] + ";}")
    html.append(".tab-content {border: 1px solid " + DASHBOARD_THEME['border'] + "; border-top: none; padding: 20px; background-color: " + DASHBOARD_THEME['background'] + ";}")
    html.append(".tab-pane {display: none;}")
    html.append("</style>")
    html.append('<div class="dashboard-container">')
    html.append('<h1>Kunden-Segmentierung & Perk-Zuweisung: Projekt-Dashboard</h1>')
    html.append('<div class="tab-buttons">')

    tab_titles = {
        1: "Executive Summary", 2: "Methodik & Datenbasis", 3: "Muster-Übersicht (ROI)",
        4: PROFILES[3]["Titel"].split(":")[1].strip(), 5: PROFILES[4]["Titel"].split(":")[1].strip(),
        6: PROFILES[0]["Titel"].split(":")[1].strip(), 7: PROFILES[(1, 2)]["Titel"].split(":")[1].strip(),
        8: "Strategische Zuweisung", 9: "Ausblick & Implementierung"
    }
    for page_num, title in tab_titles.items():
        html.append('<button class="tab-button" onclick="openTab(event, \'page' + str(page_num) + '\')">Seite ' + str(page_num) + ': ' + title + '</button>')
    html.append('</div><div class="tab-content">')

    # Page 1
    html.append('<div id="page1" class="tab-pane active">')
    html.append('<h2>Seite 1: Executive Summary & Geschaeftsziele</h2>')
    html.append('<p><strong>Geschaeftsziel:</strong> Verbesserung der Kundenbindung und des Umsatzes durch personalisierte Praemien (Perks).</p>')
    html.append('<p><strong>Methode:</strong> Unueberwachtes Lernen (K-Means Clustering) zur Identifizierung von 5 differenzierten Mustern.</p>')
    html.append('<hr><h3>Key Performance Indicators (KPIs)</h3>')
    html.append('<p><strong>Gesamtumsatz (Simuliert):</strong> ' + f"{df['total_spend'].sum():,.0f}" + ' $</p>')
    html.append('<p><strong>Anzahl Kunden im Cohort:</strong> ' + f"{df.shape[0]:,}" + '</p>')
    html.append('<hr><h3>Beispielbild: Mustergruppen nach Verhalten</h3>')
    html.append('<img src="' + IMAGE_MAP.get('gruppen_verhalten', '') + '" style="max-width: 80%; height: auto;">')
    html.append('</div>')

    # Page 2
    html.append('<div id="page2" class="tab-pane">')
    html.append('<h2>Seite 2: Methodik & Datenbasis</h2>')
    html.append('<p>Die Segmentierung basiert auf der Analyse von <strong>' + str(len(cluster_profiling.columns)) + '</strong> Kundendimensionen.</p>')
    html.append('<hr><h3>Nachweis: Korrelationen zur Feature-Auswahl</h3>')
    html.append('<img src="' + IMAGE_MAP.get('korrelation_map', '') + '" style="max-width: 100%; height: auto;">')
    html.append('</div>')

    # Page 3
    html.append('<div id="page3" class="tab-pane">')
    html.append('<h2>Seite 3: Muster-Uebersicht & Statistisches Ranking</h2>')
    html.append(profiling_html)
    html.append('</div>')

    # Pages 4-7
    for rank, cluster_key in enumerate(STRATEGIC_ORDER):
        page_num = 4 + rank
        profile = PROFILES[cluster_key]
        clusters_to_display = list(cluster_key) if isinstance(cluster_key, tuple) else [cluster_key]
        color_cluster_id = clusters_to_display[0]
        
        if not all(c in df['Cluster_M'].unique() for c in clusters_to_display):
            radar_html = "<p>Daten für dieses Cluster nicht verfügbar.</p>"
            avg_spend = 0
            avg_trips = 0
        else:
            fig = erstelle_radar_plot_figure(df, cluster_key)
            radar_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            avg_spend = df[df['Cluster_M'].isin(clusters_to_display)]['total_spend'].mean()
            avg_trips = df[df['Cluster_M'].isin(clusters_to_display)]['total_trips'].mean()
        
        html.append('<div id="page' + str(page_num) + '" class="tab-pane">')
        html.append('<h2>Seite ' + str(page_num) + ': ' + profile['Titel'] + '</h2>')
        html.append('<div style="float: right; width: 45%; padding-left: 20px;">')
        html.append('<h3>Zugewiesene Verguenstigung</h3>')
        html.append('<h2 style="color: ' + COLOR_MAP_CLUSTER_PLOTLY.get(color_cluster_id, '#000000') + '"> ' + profile['Perk'] + '</h2>')
        html.append('<p><strong>Zuweisungskriterium:</strong> ' + profile['Fokus'] + '</p>')
        html.append('<hr><h3>Wichtige Kennzahlen</h3>')
        html.append('<ul><li><strong>Durchschn. Ausgaben:</strong> ' + f"{avg_spend:,.2f}" + ' $</li><li><strong>Durchschn. Trips:</strong> ' + f"{avg_trips:.2f}" + '</li></ul>')
        html.append('</div>')
        html.append('<div style="width: 50%; float: left; height: 450px;">')
        html.append('<h3>Relatives Muster-Profil</h3>')
        html.append(radar_html)
        html.append('</div>')
        html.append('<div style="clear: both;"></div>')
        html.append('</div>')

    # Page 8
    html.append('<div id="page8" class="tab-pane">')
    html.append('<h2>Seite 8: Strategische Konsolidierung & Perk-Verteilung</h2>')
    html.append('<img src="' + IMAGE_MAP.get('kunden_verteilung', '') + '" style="max-width: 80%; height: auto;">')
    html.append('</div>')

    # Page 9
    html.append('<div id="page9" class="tab-pane">')
    html.append('<h2>Seite 9: Ausblick & Implementierung</h2>')
    html.append('<p>Neue Kunden werden basierend auf ihrem Muster abgeglichen und erhalten den entsprechenden Perk.</p>')
    html.append('</div>')

    html.append('</div>') # close tab-content
    html.append('<script>')
    html.append('function openTab(evt, pageId) {')
    html.append('var i, tabcontent, tablinks;')
    html.append('tabcontent = document.getElementsByClassName("tab-pane");')
    html.append('for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }')
    html.append('tablinks = document.getElementsByClassName("tab-button");')
    html.append('for (i = 0; i < tablinks.length; i++) { tablinks[i].className = tablinks[i].className.replace(" active", ""); }')
    html.append('document.getElementById(pageId).style.display = "block";')
    html.append('evt.currentTarget.className += " active";')
    html.append('}')
    html.append('setTimeout(() => { const firstTab = document.querySelector(".tab-button"); if (firstTab) { firstTab.click(); } }, 100);')
    html.append('</script>')
    html.append('</div>') # close dashboard-container
    
    return "".join(html)


if 'df_customer_data' in locals() and isinstance(df_customer_data, pd.DataFrame) and not df_customer_data.empty:
    final_html = generiere_dashboard_html(df_final)
    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("Dashboard wurde erfolgreich in 'dashboard.html' gespeichert.")
else:
    print("FEHLER: 'df_customer_data' wurde nicht gefunden oder ist leer.")
