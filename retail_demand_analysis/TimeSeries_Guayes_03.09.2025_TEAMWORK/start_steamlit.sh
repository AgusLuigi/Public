#!/bin/bash
# Aktiviert die Conda-Umgebung (optional, aber empfohlen)
#source /Users/cristallagus/miniconda3/bin/activate project_omni

#Öffne dein Terminal im Projektordner und gib ein:
#chmod +x run_app.sh
# Das Script starten
# Du kannst das Script jetzt jederzeit über das Terminal starten mit: ./run_app.sh
python -m streamlit run src/streamlit_app/app.py > streamlit.log 2>&1