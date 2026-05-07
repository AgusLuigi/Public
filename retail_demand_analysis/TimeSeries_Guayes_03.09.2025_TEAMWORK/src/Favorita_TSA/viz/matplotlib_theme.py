import matplotlib.pyplot as plt
from Favorita_TSA.viz.color_manager import ColorManager

def set_matplotlib_theme() -> None:
    """
    Registers the global Favorita dark theme for Matplotlib.
    Ensures brand consistency for static plots and high-volume data distributions.
    """
    # Load colors from COLORS.yaml[cite: 1, 4]
    c = ColorManager.get_colors()

    # Zuerst den Basis-Style setzen (überschreibt viele Default-Werte)
    plt.style.use('dark_background')

    # Jetzt die chirurgisch präzisen Favorita-Anpassungen vornehmen
    plt.rcParams.update({
        # 1. Backgrounds & Figure
        "figure.facecolor": c.ui_paper,      # navy_dark
        "axes.facecolor": c.ui_plot,        # charcoal
        "savefig.facecolor": c.ui_paper,
        
        # 2. Typography & Colors
        "text.color": c.ui_text,            # white_soft
        "axes.labelcolor": c.ui_text,
        "xtick.color": getattr(c, 'text_secondary', c.ui_text),
        "ytick.color": getattr(c, 'text_secondary', c.ui_text),
        "axes.titlecolor": c.ui_text,
        
        # 3. Grid & Borders[cite: 4]
        "axes.edgecolor": c.ui_border,      # deep_space
        "grid.color": c.ui_grid,            # gunmetal
        "grid.alpha": 0.4,
        "axes.grid": True,
        
        # 4. Chart Elements & Cycle[cite: 1, 4]
        "axes.prop_cycle": plt.cycler(color=c.top20), # Nutzt die Top 20 Farbliste
        "patch.edgecolor": c.ui_paper,
        "patch.force_edgecolor": True,      # Wichtig für Histogramme
        "lines.linewidth": 2,
        "lines.markersize": 6,
        
        # 5. Legend
        "legend.facecolor": c.ui_plot,
        "legend.edgecolor": c.ui_border,
        "legend.framealpha": 0.8,
        
        # 6. Spacing & Optimization
        "figure.autolayout": True,           # Verhindert abgeschnittene Labels
        "figure.dpi": 100
    })

    print("🎨 Matplotlib Theme 'favorita_dark' erfolgreich geladen.")

#set_matplotlib_theme()