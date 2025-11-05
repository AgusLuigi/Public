
#  ml-clustering-unsupervised

This project focuses on applying various clustering algorithms to an unlabeled dataset to discover inherent patterns and groupings. The goal is to perform unsupervised learning to segment data points into distinct clusters based on their features.




#  🐍 Python Projekt Setup Guide



##  📋 Requirements Management



###  🔄 Requirements.txt immer aktuell halten



**Immer wenn du ein neues Paket installierst, musst du die `requirements.txt` aktualisieren!**



##  ⚡ Schnellbefehl für neue Pakete



```bash

# Paket installieren + requirements.txt in EINEM Schritt aktualisieren

pip  install  paketname && pip  freeze  >  requirements.txt
```
## 🎯 Beispiele

```bash

# Einzelnes Paket
pip install requests && pip freeze > requirements.txt

# Mehrere Pakete
pip install pandas numpy matplotlib && pip freeze > requirements.txt

# Mit bestimmter Version
pip install flask==2.3.0 && pip freeze > requirements.txt

```
## 💡 Einzeiler für häufige Nutzung

**Alias erstellen** (optional, aber praktisch):

```bash

# In die ~/.bashrc oder ~/.zshrc eintragen:
pipin() { pip install $1 && pip freeze > requirements.txt; }

# Dann einfach:
pipin paketname
```

## 📝 Manuelle Methode (falls benötigt)

```bash

# 1. Virtuelle Umgebung aktivieren
source .venv/bin/activate
# oder auf Windows: .venv\Scripts\activate

# 2. Paket installieren
pip install paketname

# 3. Requirements.txt aktualisieren
pip freeze > requirements.txt
```

## 🔍 Überprüfung

```bash

# Inhalt der requirements.txt anzeigen
cat requirements.txt

# Test-Installation prüfen
pip install -r requirements.txt
```

## ❗ Wichtige Regeln

-   ✅ **Immer nach Paket-Installation** requirements.txt aktualisieren

-   ✅ **Nur in aktivierter virtueller Umgebung** arbeiten

-   ❌ **.venv/ Ordner nicht** in Versionskontrolle aufnehmen

## 🎯 Alternative: Direkt in Terminal hinzufügen

```bash

# Einzeiler für schnelles Hinzufügen
echo -e '\n# Pip Install mit Auto-Update requirements.txt\npipin() {\n    pip install $1 && pip freeze > requirements.txt\n    echo "✅ Paket \"$1\" installiert und requirements.txt aktualisiert!"\n}' >> ~/.zshrc && source ~/.zshrc

```

## 🔍 Überprüfen ob es funktioniert:

```bash

# Funktion testen
pipin --help

# Oder prüfen ob Funktion existiert
type pipin
```
**So bleibt dein Projekt immer synchronisiert!** ✅