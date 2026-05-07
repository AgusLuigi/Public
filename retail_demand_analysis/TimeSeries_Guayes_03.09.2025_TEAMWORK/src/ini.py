def initialisation():
    """
    Initialisiert das Projekt autonom:
    1. Findet den Root-Pfad dynamisch (Stochastische Suche).
    2. Installiert fehlende externe Bibliotheken automatisch (Auto-Pip).
    3. Importiert interne Module und injiziert sie ins Notebook.
    4. Aktiviert Designs mit Fallback-Logik für fehlende Farb-Attribute.
    """
    import os
    import sys
    import subprocess
    import importlib
    import pkgutil
    from pathlib import Path
    import __main__ 

    # 1. DYNAMISCHE ROOT-ERKENNUNG
    current_path = Path.cwd()
    root_path = next((p for p in [current_path] + list(current_path.parents) 
                    if (p / "src").exists() or (p / ".git").exists()), current_path)
    
    src_path = str((root_path / "src").absolute())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    def install_and_import(pkg_name):
        """Installiert ein Paket falls nötig und gibt das Modul zurück."""
        if pkg_name.startswith("Favorita_TSA"):
            try:
                return importlib.import_module(pkg_name)
            except ImportError:
                return None
        try:
            return importlib.import_module(pkg_name)
        except ImportError:
            base_pkg = pkg_name.split('.')[0]
            print(f"📦 Paket '{base_pkg}' fehlt. Starte autonome Installation...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", base_pkg])
                return importlib.import_module(pkg_name)
            except Exception as e:
                print(f"⚠️ Installation von {base_pkg} fehlgeschlagen: {e}")
                return None

    try:
        # 2. AUTOMATISIERTE EXTERN-INJEKTION (Mit statsmodels Erweiterung)
        external_map = {
            'pandas': 'pd',
            'plotly.express': 'px',
            'matplotlib.pyplot': 'plt',
            'numpy': 'np',
            'seaborn': 'sns',
            'mlflow': 'ml',
            'statsmodels.api': 'sm'
        }
        
        injections = {'os': os, 'Path': Path}
        
        for pkg_full_name, alias in external_map.items():
            mod = install_and_import(pkg_full_name)
            if mod:
                injections[alias] = mod

        # 3. AUTONOMER IMPORT (Favorita_TSA) MIT REKURSIVEM SCAN
        try:
            import Favorita_TSA
            for sub_pkg in ['preprocess', 'models', 'viz', 'utils']:
                package_path = f"Favorita_TSA.{sub_pkg}"
                try:
                    package = importlib.import_module(package_path)
                    # CHIRURGISCHES UPGRADE: walk_packages für tiefe Modul-Strukturen
                    for info in pkgutil.walk_packages(package.__path__, package_path + "."):
                        try:
                            module = importlib.import_module(info.name)
                            for attr_name in dir(module):
                                if not attr_name.startswith('_'):
                                    injections[attr_name] = getattr(module, attr_name)
                        except ImportError as e:
                            missing = str(e).split("'")[-2]
                            install_and_import(missing)
                            # Zweiter Versuch nach Installation
                            module = importlib.import_module(info.name)
                            for attr_name in dir(module):
                                if not attr_name.startswith('_'):
                                    injections[attr_name] = getattr(module, attr_name)
                except: continue
        except: pass

        # 4. CONFIG & THEME AUTO-START (Mit expliziter 'colors' Injektion)
        colors_path = root_path / "configs" / "COLORS.yaml"
        if not colors_path.exists():
            colors_path = root_path / "src" / "configs" / "COLORS.yaml"

        if colors_path.exists() and 'ColorManager' in injections:
            # Nutzt den ColorManager zum Laden der YAML
            c = injections['ColorManager'].get_colors(file_path=str(colors_path))
            
            # CHIRURGISCHE INJEKTION: Stellt 'colors' direkt im Notebook bereit
            injections['colors'] = c 
            
            # Dynamischer Fallback für UI-Elemente
            if not hasattr(c, 'text_secondary'):
                fallback = getattr(c, 'gray_steel', '#A9A9A9')
                setattr(c, 'text_secondary', fallback)

            if 'set_plotly_theme' in injections: injections['set_plotly_theme']()
            if 'set_matplotlib_theme' in injections: injections['set_matplotlib_theme']()
            status = "✅ System bereit (incl. Colors & Deep-Scan)"
        else:
            status = "⚠️ System bereit (Themes eingeschränkt)"

        # --- FINALE INJEKTION IN DAS NOTEBOOK ---
        for name, obj in injections.items():
            setattr(__main__, name, obj)

        # 5. AUDIT
        print(f"{status}! Alles wurde autonom geladen.")
        print(f"   -> Root: {root_path}")
        print(f"   -> Aliase: {', '.join([k for k in injections if len(k) <= 3])}")
        
        return root_path

    except Exception as e:
        print(f"❌ Kritischer Fehler bei der Autonomie-Initialisierung: {e}")
        return None
    
# Start
root_path = initialisation()