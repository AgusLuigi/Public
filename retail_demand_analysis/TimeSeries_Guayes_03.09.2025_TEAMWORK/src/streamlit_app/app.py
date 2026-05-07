import sys
from pathlib import Path

# --- 1. DYNAMISCHE PFAD-LOGIK (Aufwärtssuche) ---
CURRENT_FILE = Path(__file__).resolve()
ROOT = CURRENT_FILE.parents[2] 

SRC_PATH = str(ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

for parent in CURRENT_FILE.parents:
    if str(parent) not in sys.path:
        sys.path.append(str(parent))

import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

# Import der Themes mit Favorita-Struktur
try:
    from Favorita_TSA.viz.ploty_theme import set_plotly_theme
    from Favorita_TSA.viz.streamlit_theme import get_glass_theme
    set_plotly_theme()
except ImportError:
    try:
        from streamlit_theme import get_glass_theme
    except ImportError:
        def get_glass_theme(mode="dark"):
            if mode == "dark":
                return {
                    "mode": "dark", "bg_color": "#05070a", "text_color": "#FFFFFF", 
                    "panel_bg": "rgba(10, 15, 25, 0.7)", "accent": "#C0C0C0", 
                    "panel_border": "rgba(255,255,255,0.1)", "glass_blur": "30px", 
                    "shadow": "rgba(0,0,0,0.9)", "brand_gradient": "linear-gradient(135deg, #a78bfa, #22d3ee)"
                }
            return {
                "mode": "light", "bg_color": "#f8fafc", "text_color": "#0f172a", 
                "panel_bg": "rgba(255, 255, 255, 0.7)", "accent": "#D4AF37", 
                "panel_border": "rgba(0,0,0,0.05)", "glass_blur": "25px", 
                "shadow": "rgba(0,0,0,0.1)", "brand_gradient": "linear-gradient(135deg, #D4AF37, #facc15)"
            }

# --- 2. CONFIG ---
st.set_page_config(
    page_title="Favorita TSA — Futuristic System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

# --- 3. SIDEBAR (Command Center) ---
with st.sidebar:
    st.title("Settings")
    toggle = st.toggle("Gold Protocol (Light Mode)", value=(st.session_state.theme_mode == "light"))
    st.session_state.theme_mode = "light" if toggle else "dark"
    st.markdown("---")
    st.title("Navigation")
    st.info("Work Redy. Render Engine: ACTIVE")

theme = get_glass_theme(st.session_state.theme_mode)
pio.templates.default = "favorita_dark" if theme.get("mode") == "dark" else "plotly_white"

# --- 4. CSS (FUTURISTIC UI & FIXES) ---
st.markdown(
    f"""
    <style>
    /* Hintergrund-IFrame fixieren */
    iframe[title="st.component_browser.v1.html_component"] {{
        position: fixed !important; top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: -1 !important; pointer-events: none !important; border: none;
    }}
    
    .stApp {{ background-color: transparent !important; }}
    
    /* Text-Synchronisation */
    .stApp, .stMarkdown, p, h1, h2, h3, span, label, .stCaption {{
        color: {theme['text_color']} !important;
        text-shadow: 0 2px 4px {theme['shadow']};
    }}

    /* Sidebar Design mit Blur */
    section[data-testid="stSidebar"] {{
        background-color: {theme['bg_color']}CC !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid {theme['panel_border']};
        z-index: 100;
    }}

    header[data-testid="stHeader"] {{ background: transparent !important; }}

    /* Sidebar-Chevron & Header Buttons Fix */
    button[data-testid="stSidebarCollapse"] svg, 
    button[kind="headerNoContext"] svg {{
        fill: {theme['accent']} !important;
        stroke: {theme['accent']} !important;
    }}
    
    button[data-testid="stSidebarCollapse"],
    button[kind="headerNoContext"] {{
        background-color: {theme['panel_bg']} !important;
        border: 1px solid {theme['panel_border']} !important;
        z-index: 9999 !important;
    }}

    /* Öffnen-Pfeil (Collapsed Zustand) */
    .st-emotion-cache-hp888a {{ color: {theme['accent']} !important; }}

    .block-container {{ padding: 0 !important; max-width: 100% !important; }}
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 5. 4K FUTURISTIC 3D ENGINE (SHARP STARS, PLANET, GROCERY SHARDS) ---
HTML_CONTENT = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<style>
html, body {{ margin: 0; height: 100%; overflow: hidden; font-family: Inter, sans-serif; background: {theme['bg_color']}; }}
#bg {{ position: fixed; inset: 0; z-index: 0; }}
.stage {{ position: relative; z-index: 10; height: 100vh; display: grid; place-items: center; pointer-events: none; }}
.panel {{
  pointer-events: auto;
  width: min(980px, 92vw); padding: 78px 88px; border-radius: 40px;
  background: {theme['panel_bg']};
  backdrop-filter: blur({theme['glass_blur']});
  border: 1px solid {theme['panel_border']};
  box-shadow: 0 90px 220px {theme['shadow']};
  text-align: left; opacity: 0; transform: translateY(20px) scale(0.97);
  color: {theme['text_color']};
}}
.panel::before {{
  content: ""; position: absolute; inset: -2px; border-radius: inherit;
  background: conic-gradient(from 160deg, transparent, {theme['accent']}88, #22d3ee88, #4ade8088, transparent);
  filter: blur(18px); opacity: 0.35; animation: ring 14s linear infinite; z-index: -1;
}}
@keyframes ring {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
.kicker {{ color: {theme['text_color']}; opacity: 0.6; font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; }}
h1 {{ margin: 16px 0 0 0; font-size: clamp(48px, 6vw, 80px); font-weight: 600; line-height: 1.05; }}
.grad {{ background: {theme['brand_gradient']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.sub {{ margin-top: 20px; max-width: 62ch; font-size: 18px; line-height: 1.65; opacity: 0.7; }}
.meta {{ margin-top: 34px; font-size: 14px; opacity: 0.5; }}
</style>
</head>
<body>
<canvas id="bg"></canvas>
<div class="stage">
  <div class="panel" id="panel">
    <div class="kicker">RETAIL TIME-SERIES FORECASTING · ML SYSTEM</div>
    <h1><span class="grad">Favorita TSA</span></h1>
    <div class="sub">End-to-end machine learning pipeline for grocery sales forecasting. Time-series analysis.</div>
    <div class="meta">Patrick · Agus · Kiko</div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/animejs@3.2.2/lib/anime.min.js"></script>
<script>
const canvas = document.getElementById("bg");
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, alpha: true }});
renderer.setPixelRatio(window.devicePixelRatio > 1 ? 2 : 1);
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.1, 1000);
camera.position.z = 15;

const group = new THREE.Group();
scene.add(group);

/* BELEUCHTUNG */
scene.add(new THREE.AmbientLight(0xffffff, 0.2));
const sun = new THREE.DirectionalLight('{theme['accent']}', 2);
sun.position.set(5, 10, 7);
scene.add(sun);

/* SHARP STARS */
const starGeo = new THREE.BufferGeometry();
const starPos = new Float32Array(5000 * 3);
for(let i=0; i<15000; i++) starPos[i] = (Math.random()-0.5)*300;
starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({{ size: 0.15, color: 0xffffff, transparent: true, opacity: 0.8 }}));
scene.add(stars);

/* CORE PLANET */
const planet = new THREE.Mesh(
  new THREE.IcosahedronGeometry(4, 15),
  new THREE.MeshStandardMaterial({{ color: 0x111827, wireframe: true, transparent: true, opacity: 0.2 }})
);
group.add(planet);

const innerPlanet = new THREE.Mesh(
    new THREE.SphereGeometry(3.8, 64, 64),
    new THREE.MeshStandardMaterial({{ color: 0x030712, emissive: '{theme['accent']}', emissiveIntensity: 0.05 }})
);
group.add(innerPlanet);

/* FOOD SHARDS */
const shards = new THREE.Group();
const shardGeo = new THREE.OctahedronGeometry(0.3, 0);
for(let i=0; i<40; i++) {{
    const mat = new THREE.MeshStandardMaterial({{ color: i%2===0 ? 0x22d3ee : 0xa78bfa, emissiveIntensity: 0.5, transparent: true, opacity: 0.8 }});
    const m = new THREE.Mesh(shardGeo, mat);
    const a = Math.random()*Math.PI*2;
    const r = 6 + Math.random()*4;
    m.position.set(Math.cos(a)*r, (Math.random()-0.5)*10, Math.sin(a)*r);
    shards.add(m);
}}
group.add(shards);

anime({{ targets: "#panel", opacity: [0, 1], translateY: [20, 0], scale: [0.97, 1], duration: 1500, easing: "easeOutExpo" }});

const mouse = {{ x: 0, y: 0 }};
window.addEventListener("mousemove", e => {{ mouse.x=(e.clientX/innerWidth)*2-1; mouse.y=-(e.clientY/innerHeight)*2+1; }});

function tick() {{
    group.rotation.y += 0.001;
    shards.rotation.y += 0.002;
    group.position.x += (mouse.x*0.5 - group.position.x)*0.02;
    group.position.y += (mouse.y*0.5 - group.position.y)*0.02;
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
}}
tick();

window.addEventListener("resize", () => {{
    renderer.setSize(innerWidth, innerHeight);
    camera.aspect = innerWidth/innerHeight;
    camera.updateProjectionMatrix();
}});
</script>
</body>
</html>
"""

def main():
    components.html(HTML_CONTENT, height=1200, scrolling=False)
    st.write("### System Operational")
    st.caption(f"Sync: 4K Ultra | Mode: {st.session_state.theme_mode.upper()} | Path: Dynamic")

if __name__ == "__main__":
    main()
