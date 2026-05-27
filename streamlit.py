"""
Fractal Generator — Mandelbrot & Julia Sets (Streamlit version)
===============================================================
Requirements: pip install streamlit numpy pillow
Run with:     streamlit run fractal_generator_streamlit.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import io

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fractal Generator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #0a0a0f;
    color: #c8a8ff;
}
section[data-testid="stSidebar"] {
    background-color: #0d0d18;
    border-right: 1px solid #1e1e30;
}
section[data-testid="stSidebar"] * {
    font-family: 'Share Tech Mono', monospace !important;
    color: #c8a8ff !important;
}
h1, h2, h3 { color: #c8a8ff !important; }
.stButton > button {
    background-color: #1a1a2e;
    color: #c8a8ff;
    border: 1px solid #3a2a5e;
    font-family: 'Share Tech Mono', monospace;
    width: 100%;
}
.stButton > button:hover {
    background-color: #2a1a4e;
    color: #ffffff;
    border-color: #c8a8ff;
}
.stDownloadButton > button {
    background-color: #1a1a2e;
    color: #c8a8ff;
    border: 1px solid #3a2a5e;
    font-family: 'Share Tech Mono', monospace;
    width: 100%;
}
.stDownloadButton > button:hover {
    background-color: #2a1a4e;
    color: #ffffff;
}
.stSelectbox > div, .stRadio > div {
    background-color: #0d0d18;
}
div[data-testid="stMarkdownContainer"] p {
    color: #7a7a9a;
    font-size: 0.8rem;
}
.stSlider > div > div > div {
    background-color: #c8a8ff;
}
</style>
""", unsafe_allow_html=True)


# ── Colour maps ──────────────────────────────────────────────────────────────

def make_colormap(name: str) -> np.ndarray:
    t = np.linspace(0, 1, 256)
    if name == "Ultra":
        r = np.clip(np.sin(t * np.pi * 3) * 0.5 + 0.5, 0, 1)
        g = np.clip(np.sin(t * np.pi * 3 + 2.1) * 0.5 + 0.5, 0, 1)
        b = np.clip(np.sin(t * np.pi * 3 + 4.2) * 0.5 + 0.5, 0, 1)
    elif name == "Fire":
        r = np.clip(t * 3, 0, 1)
        g = np.clip(t * 3 - 1, 0, 1)
        b = np.clip(t * 3 - 2, 0, 1)
    elif name == "Ice":
        r = t ** 2
        g = t
        b = np.ones(256)
    elif name == "Gold":
        r = np.sqrt(t)
        g = t ** 1.5
        b = t ** 3
    elif name == "Psychedelic":
        r = 0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.0))
        g = 0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.33))
        b = 0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.67))
    else:  # Classic
        r = t
        g = t * 0.6
        b = 1 - t
    lut = np.stack([r, g, b], axis=1)
    return (np.clip(lut, 0, 1) * 255).astype(np.uint8)


COLORMAPS = ["Ultra", "Fire", "Ice", "Gold", "Psychedelic", "Classic"]


# ── Fractal computation ──────────────────────────────────────────────────────

def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width, dtype=np.float64)
    y = np.linspace(ymin, ymax, height, dtype=np.float64)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    Z = np.zeros_like(C)
    count = np.zeros(C.shape, dtype=np.float64)
    mask = np.ones(C.shape, dtype=bool)
    for i in range(1, max_iter + 1):
        Z[mask] = Z[mask] ** 2 + C[mask]
        escaped = mask & (np.abs(Z) > 2)
        count[escaped] = i - np.log2(np.log2(np.abs(Z[escaped]) + 1e-10))
        mask[escaped] = False
    return count, mask


def compute_julia(xmin, xmax, ymin, ymax, width, height, max_iter, c):
    x = np.linspace(xmin, xmax, width, dtype=np.float64)
    y = np.linspace(ymin, ymax, height, dtype=np.float64)
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    count = np.zeros(Z.shape, dtype=np.float64)
    mask = np.ones(Z.shape, dtype=bool)
    for i in range(1, max_iter + 1):
        Z[mask] = Z[mask] ** 2 + c
        escaped = mask & (np.abs(Z) > 2)
        count[escaped] = i - np.log2(np.log2(np.abs(Z[escaped]) + 1e-10))
        mask[escaped] = False
    return count, mask


def render_to_image(count, mask, lut, max_iter):
    h, w = count.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    outside = ~mask
    if outside.any():
        vals = count[outside]
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            norm = np.clip((vals - vmin) / (vmax - vmin), 0, 1)
        else:
            norm = np.zeros_like(vals)
        idx = (norm * 255).astype(np.uint8)
        rgb[outside] = lut[idx]
    return Image.fromarray(rgb, "RGB")


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Session state defaults ───────────────────────────────────────────────────

def _init_state():
    defaults = {
        "xmin": -2.5, "xmax": 1.0,
        "ymin": -1.4, "ymax": 1.4,
        "mode": "Mandelbrot",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar controls ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ✦ FRACTAL\n## GENERATOR")
    st.markdown("*Mandelbrot · Julia Sets*")
    st.divider()

    mode = st.radio("FRACTAL TYPE", ["Mandelbrot", "Julia"],
                    index=0 if st.session_state.mode == "Mandelbrot" else 1)

    if mode != st.session_state.mode:
        st.session_state.mode = mode
        if mode == "Mandelbrot":
            st.session_state.xmin, st.session_state.xmax = -2.5, 1.0
            st.session_state.ymin, st.session_state.ymax = -1.4, 1.4
        else:
            st.session_state.xmin, st.session_state.xmax = -2.0, 2.0
            st.session_state.ymin, st.session_state.ymax = -1.6, 1.6

    st.divider()

    st.markdown("**JULIA CONSTANT  c = a + bi**")
    julia_re = st.slider("Re(c)", -2.0, 2.0, -0.7, 0.001)
    julia_im = st.slider("Im(c)", -2.0, 2.0,  0.27, 0.001)
    julia_c  = complex(julia_re, julia_im)
    sign = "+" if julia_im >= 0 else ""
    st.markdown(f"`c = {julia_re:+.4f} {sign} {abs(julia_im):.4f}i`")

    st.divider()

    max_iter = st.select_slider("MAX ITERATIONS", options=[100, 200, 500, 1000], value=200)
    colormap = st.selectbox("COLOUR MAP", COLORMAPS, index=0)

    st.divider()

    st.markdown("**VIEW RANGE**")
    col1, col2 = st.columns(2)
    with col1:
        xmin = st.number_input("Re min", value=st.session_state.xmin, step=0.1, format="%.3f")
        ymin = st.number_input("Im min", value=st.session_state.ymin, step=0.1, format="%.3f")
    with col2:
        xmax = st.number_input("Re max", value=st.session_state.xmax, step=0.1, format="%.3f")
        ymax = st.number_input("Im max", value=st.session_state.ymax, step=0.1, format="%.3f")

    if st.button("⟳  RESET VIEW"):
        if mode == "Mandelbrot":
            xmin, xmax, ymin, ymax = -2.5, 1.0, -1.4, 1.4
        else:
            xmin, xmax, ymin, ymax = -2.0, 2.0, -1.6, 1.6
        st.session_state.xmin, st.session_state.xmax = xmin, xmax
        st.session_state.ymin, st.session_state.ymax = ymin, ymax
        st.rerun()

    st.divider()
    st.markdown("*Tip: Zoom into interesting areas by adjusting Re/Im range manually.*")


# ── Main render ──────────────────────────────────────────────────────────────

st.markdown(f"### ✦ {mode} Set")
st.markdown(f"`Re: [{xmin:.4f}, {xmax:.4f}]  Im: [{ymin:.4f}, {ymax:.4f}]  iters: {max_iter}  cmap: {colormap}`")

with st.spinner("Rendering fractal…"):
    lut = make_colormap(colormap)
    if mode == "Mandelbrot":
        count, mask = compute_mandelbrot(xmin, xmax, ymin, ymax, 900, 600, max_iter)
    else:
        count, mask = compute_julia(xmin, xmax, ymin, ymax, 900, 600, max_iter, julia_c)

    img = render_to_image(count, mask, lut, max_iter)

st.image(img, use_container_width=True)

png_bytes = image_to_bytes(img)
st.download_button(
    label="💾  SAVE PNG",
    data=png_bytes,
    file_name=f"{mode.lower()}_fractal.png",
    mime="image/png",
)
