import sys
from pathlib import Path

import plotly.io as pio
import streamlit as st

from Favorita_TSA.viz.ploty_theme import set_plotly_theme

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

set_plotly_theme()
pio.templates.default = "favorita_dark"

st.set_page_config(
    page_title="Favorita TSA — System",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- FULLSCREEN STREAMLIT ----------
st.markdown(
    """
    <style>
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }

    footer {visibility: hidden;}

    iframe {
        display: block;
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
html, body {
  margin: 0;
  height: 100%;
  overflow: hidden;
  font-family: Inter, system-ui, -apple-system, sans-serif;
  background: transparent;
}

#bg {
  position: fixed;
  inset: 0;
  z-index: 0;
}

.stage {
  position: relative;
  z-index: 10;
  height: 100vh;
  display: grid;
  place-items: center;
  padding: 32px;
}

.panel {
  width: min(980px, 92vw);
  padding: 78px 88px;
  border-radius: 30px;
  background: rgba(255,255,255,0.055);;
  backdrop-filter: blur(26px);
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 90px 220px rgba(0,0,0,0.92);
  overflow: hidden;
  opacity: 0;
  transform: translateY(18px) scale(0.97);
}

.panel::before {
  content: "";
  position: absolute;
  inset: -2px;
  border-radius: inherit;
  background: conic-gradient(from 160deg,
    rgba(167,139,250,0.0),
    rgba(167,139,250,0.55),
    rgba(34,211,238,0.55),
    rgba(74,222,128,0.55),
    rgba(250,204,21,0.45),
    rgba(167,139,250,0.55),
    rgba(167,139,250,0.0)
  );
  filter: blur(18px);
  opacity: 0.35;
  animation: ring 14s linear infinite;
  z-index: -1;
}

@keyframes ring {
  from { transform: rotate(0deg);}
  to   { transform: rotate(360deg);}
}

.kicker {
  color: rgba(255,255,255,0.62);
  font-size: 13px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

h1 {
  margin: 16px 0 0 0;
  font-size: clamp(48px, 6vw, 80px);
  font-weight: 600;
  letter-spacing: -0.05em;
  line-height: 1.05;
  color: rgba(255,255,255,0.97);
}

.grad {
  background: linear-gradient(120deg, #a78bfa, #22d3ee, #4ade80);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.sub {
  margin-top: 20px;
  max-width: 62ch;
  font-size: 18px;
  line-height: 1.65;
  color: rgba(255,255,255,0.62);
}

.meta {
  margin-top: 34px;
  font-size: 14px;
  color: rgba(255,255,255,0.42);
}
</style>
</head>

<body>
<canvas id="bg"></canvas>

<div class="stage">
  <div class="panel" id="panel">
    <div class="kicker">RETAIL TIME-SERIES FORECASTING · ML SYSTEM</div>
    <h1><span class="grad">Favorita TSA</span></h1>
    <div class="sub">
      End-to-end machine learning pipeline for grocery sales forecasting.
      Temporal feature engineering, promotion effects and store dynamics
      modeled through scalable time-series architecture.
    </div>
    <div class="meta">Patrick · Agus · Kiko</div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/animejs@3.2.2/lib/anime.min.js"></script>

<script>
/* ====== STREAMLIT DARK BASE ====== */
function getStreamlitBg() {
  try {
    const root = window.parent.document.documentElement;
    const styles = getComputedStyle(root);
    const c = styles.getPropertyValue("--background-color").trim();
    if (c) return c;
  } catch {}
  return "#0e1117"; // exact Streamlit dark
}

const bgColor = getStreamlitBg();

/* soften to gray-black tone */
const fogColor = bgColor;
document.body.style.background = bgColor;

/* ====== RENDERER ====== */
const canvas = document.getElementById("bg");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(fogColor, 1);

/* ====== SCENE ====== */
const scene = new THREE.Scene();

renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 100);
camera.position.set(0, 0.2, 10.5);

const group = new THREE.Group();
scene.add(group);

scene.add(new THREE.AmbientLight(0xcfd6e6, 0.35));

const key = new THREE.PointLight(0xa78bfa, 1.0, 40);
key.position.set(5, 4, 10);
scene.add(key);

const fill = new THREE.PointLight(0x22d3ee, 0.7, 40);
fill.position.set(-6, -2, 8);
scene.add(fill);

const rim = new THREE.PointLight(0x4ade80, 0.6, 40);
rim.position.set(0, 6, -6);
scene.add(rim);

/* stars */
const starCount = 1400;
const starGeo = new THREE.BufferGeometry();
const starPos = new Float32Array(starCount * 3);
for (let i = 0; i < starCount; i++) {
  const ix = i * 3;
  starPos[ix] = (Math.random() - 0.5) * 60;
  starPos[ix+1] = (Math.random() - 0.5) * 36;
  starPos[ix+2] = -Math.random() * 60;
}
starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
const stars = new THREE.Points(
  starGeo,
  new THREE.PointsMaterial({ size: 0.03, color: 0xffffff, transparent: true, opacity: 0.45 })
);
scene.add(stars);

/* core */
const core = new THREE.Mesh(
  new THREE.TorusKnotGeometry(1.2, 0.34, 220, 18, 2, 3),
  new THREE.MeshStandardMaterial({
    color: 0x161b22,
    metalness: 0.55,
    roughness: 0.22,
    emissive: 0x0b0f14,
    emissiveIntensity: 0.7
  })
);
group.add(core);

const glow = new THREE.Mesh(
  new THREE.SphereGeometry(1.95, 48, 48),
  new THREE.MeshBasicMaterial({
    color: 0x8b7cff,
    transparent: true,
    opacity: 0.06,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    depthTest: false
  })
);
group.add(glow);

/* rings */
function ring(r,c,o){
  return new THREE.Mesh(
    new THREE.TorusGeometry(r,0.02,10,300),
    new THREE.MeshBasicMaterial({ color:c, transparent:true, opacity:o, blending:THREE.AdditiveBlending,   depthWrite:false,
  depthTest:false })
  );
}
const r1 = ring(2.55,0x22d3ee,0.14); r1.rotation.x=Math.PI*0.5;
const r2 = ring(2.95,0xa78bfa,0.10); r2.rotation.y=Math.PI*0.35;
const r3 = ring(3.35,0x4ade80,0.08); r3.rotation.z=Math.PI*0.22;
group.add(r1,r2,r3);

/* shards */
const shardGeo = new THREE.BoxGeometry(0.05,0.35,0.05);
const shardMat = new THREE.MeshBasicMaterial({ color:0xffffff, transparent:true, opacity:0.55, blending:THREE.AdditiveBlending });
const shards = new THREE.InstancedMesh(shardGeo, shardMat, 90);
const dummy = new THREE.Object3D();
const cols = [0xa78bfa,0x22d3ee,0x4ade80,0xfacc15];

for(let i=0;i<90;i++){
  const a = Math.random()*Math.PI*2;
  const y = (Math.random()-0.5)*2.2;
  const r = 3.8+Math.random()*1.6;
  dummy.position.set(Math.cos(a)*r,y,Math.sin(a)*r);
  dummy.rotation.set(Math.random()*Math.PI,Math.random()*Math.PI,Math.random()*Math.PI);
  dummy.scale.setScalar(0.8+Math.random()*1.4);
  dummy.updateMatrix();
  shards.setMatrixAt(i,dummy.matrix);
  shards.setColorAt(i,new THREE.Color(cols[i%cols.length]));
}
group.add(shards);

/* mouse parallax */
const mouse={x:0,y:0};
addEventListener("mousemove",e=>{
  mouse.x=(e.clientX/innerWidth)*2-1;
  mouse.y=-((e.clientY/innerHeight)*2-1);
});

/* panel intro */
anime({
  targets:"#panel",
  opacity:[0,1],
  translateY:[18,0],
  scale:[0.97,1],
  duration:1200,
  easing:"easeOutExpo"
});

/* loop */
function tick(t){
  const time=t*0.001;

  const px=mouse.x*0.25;
  const py=mouse.y*0.18;

  group.rotation.y+=0.0022;
  group.rotation.x+=0.0012;

  group.position.x+= (px-group.position.x)*0.05;
  group.position.y+= (py-group.position.y)*0.05;

  r1.rotation.z+=0.0022;
  r2.rotation.x+=0.0016;
  r3.rotation.y+=0.0019;

  core.material.emissiveIntensity=0.7+0.35*Math.sin(time*1.2);

  stars.rotation.y+=0.00035;
  stars.rotation.x+=0.00015;

  renderer.render(scene,camera);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

addEventListener("resize",()=>{
  renderer.setSize(innerWidth,innerHeight);
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
});
</script>
</body>
</html>
"""

st.components.v1.html(HTML, height=1200, scrolling=False)
