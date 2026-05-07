class ColorFallback:
    black_rich = "#0e1117"
    ui_text = "#FFFFFF"
    blue_bright = "#22d3ee"

def get_glass_theme(mode="dark"):
    """Zentrale Sammlung für Fenster-Logik: Dark (Silber) vs Light (Gold)"""
    c = ColorFallback()

    def hex_to_rgba(hex_color, opacity):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"

    if mode == "dark":
        accent = "#C0C0C0"  # Silber
        bg_main = c.black_rich
        panel_bg = "rgba(255, 255, 255, 0.01)" 
        border_color = hex_to_rgba(accent, 0.2)
        text_color = c.ui_text
        shadow = "rgba(0,0,0,0.8)"
        blur = "12px"
    else:
        accent = "#D4AF37"  # Gold
        bg_main = "#F0F2F6" 
        panel_bg = "rgba(255, 255, 255, 0.4)"
        border_color = hex_to_rgba(accent, 0.4)
        text_color = "#1C1C1C"
        shadow = "rgba(0,0,0,0.1)"
        blur = "8px"

    return {
        "mode": mode,
        "bg_color": bg_main,
        "panel_bg": panel_bg,
        "panel_border": border_color,
        "glass_blur": blur,
        "text_color": text_color,
        "accent": accent,
        "shadow": shadow,
        "brand_gradient": f"linear-gradient(135deg, {c.blue_bright}, {accent})"
    }