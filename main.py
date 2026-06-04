import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

MODEL_PATH = Path(__file__).parent / "best_cnn.h5"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (224, 224)

CLASS_INFO = {
    "glioma": {"label": "Gliome", "desc": "Tumeur du tissu glial", "color": "#ef4444"},
    "meningioma": {"label": "Méningiome", "desc": "Tumeur des méninges", "color": "#f59e0b"},
    "notumor": {"label": "Aucune tumeur", "desc": "Aucune tumeur détectée", "color": "#22c55e"},
    "pituitary": {"label": "Hypophyse", "desc": "Tumeur de l'hypophyse", "color": "#3b82f6"},
}

st.set_page_config(
    page_title="Brain Tumor Classifier",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.15), transparent 35%),
        radial-gradient(circle at bottom right, rgba(34,197,94,0.10), transparent 30%),
        #080b12;
    color: #f8fafc;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
}

.hero {
    padding: 2rem;
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 28px;
    background: rgba(15,23,42,0.72);
    box-shadow: 0 24px 70px rgba(0,0,0,0.35);
    backdrop-filter: blur(18px);
    margin-bottom: 1.6rem;
}

.hero h1 {
    font-size: 3rem;
    line-height: 1;
    margin: 0;
    letter-spacing: -0.06em;
}

.hero p {
    margin-top: 0.8rem;
    color: #94a3b8;
    font-size: 1rem;
}

.badges {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}

.badge {
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.25);
    color: #bfdbfe;
    font-size: 0.82rem;
    font-weight: 600;
}

.panel {
    padding: 1.25rem;
    border-radius: 24px;
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 18px 50px rgba(0,0,0,0.25);
    min-height: 100%;
}

.panel-title {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    margin-bottom: 1rem;
}

.result {
    padding: 1.4rem;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(15,23,42,0.9));
    border: 1px solid rgba(59,130,246,0.35);
    margin-bottom: 1rem;
}

.result-label {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.05em;
}

.result-desc {
    color: #94a3b8;
    margin-top: 0.35rem;
}

.confidence {
    font-size: 3.4rem;
    font-weight: 900;
    letter-spacing: -0.08em;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.8rem;
    margin-top: 1rem;
}

.metric {
    padding: 1rem;
    border-radius: 18px;
    background: rgba(2,6,23,0.55);
    border: 1px solid rgba(148,163,184,0.14);
}

.metric-name {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 1.3rem;
    font-weight: 800;
}

.bar-row {
    margin-bottom: 1rem;
}

.bar-top {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    margin-bottom: 0.45rem;
}

.bar-name {
    color: #e2e8f0;
    font-weight: 700;
}

.bar-pct {
    color: #94a3b8;
    font-weight: 700;
}

.bar-track {
    height: 10px;
    border-radius: 999px;
    background: rgba(148,163,184,0.14);
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
}

.empty {
    height: 360px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #64748b;
    text-align: center;
    border: 1px dashed rgba(148,163,184,0.22);
    border-radius: 22px;
    background: rgba(2,6,23,0.25);
}

.disclaimer {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: 18px;
    background: rgba(245,158,11,0.10);
    border: 1px solid rgba(245,158,11,0.25);
    color: #fcd34d;
    font-size: 0.85rem;
}

[data-testid="stFileUploader"] {
    padding: 1rem;
    border-radius: 22px;
    background: rgba(2,6,23,0.35);
    border: 1px dashed rgba(148,163,184,0.28);
}

.stImage img {
    border-radius: 22px;
    border: 1px solid rgba(148,163,184,0.18);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(str(MODEL_PATH.resolve()))


model = load_model()

st.markdown("""
<div class="hero">
    <h1>Brain Tumor Classifier</h1>
    <p>Analyse automatique d'IRM cérébrale avec un modèle CNN entraîné sur 4 classes.</p>
    <div class="badges">
        <div class="badge">CNN</div>
        <div class="badge">IRM cérébrale</div>
        <div class="badge">4 classes</div>
        <div class="badge">TensorFlow</div>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(f"Modèle introuvable : {MODEL_PATH}")
    st.stop()

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Image à analyser</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Importer une image IRM",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        label_visibility="collapsed"
    )

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, use_container_width=True)

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-name">Fichier</div>
                <div class="metric-value">{uploaded.name}</div>
            </div>
            <div class="metric">
                <div class="metric-name">Résolution</div>
                <div class="metric-value">{img_pil.size[0]} × {img_pil.size[1]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty">
            <div>
                Importez une image IRM pour lancer l'analyse
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Résultat du modèle</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div class="empty">
            En attente d'une image...
        </div>
        """, unsafe_allow_html=True)
    else:
        img_array = np.array(img_pil.resize(IMG_SIZE), dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyse en cours..."):
            probs = model.predict(img_batch, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        info = CLASS_INFO[pred_class]

        st.markdown(f"""
        <div class="result">
            <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;">
                <div>
                    <div class="result-label">{info["label"]}</div>
                    <div class="result-desc">{info["desc"]}</div>
                </div>
                <div class="confidence" style="color:{info["color"]};">{confidence * 100:.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="panel-title" style="margin-top:1.2rem;">Probabilités</div>', unsafe_allow_html=True)

        bars = ""
        for i, cls in enumerate(CLASS_NAMES):
            pct = float(probs[i]) * 100
            color = CLASS_INFO[cls]["color"]
            label = CLASS_INFO[cls]["label"]

            bars += f"""
            <div class="bar-row">
                <div class="bar-top">
                    <div class="bar-name">{label}</div>
                    <div class="bar-pct">{pct:.1f}%</div>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
                </div>
            </div>
            """

        st.markdown(bars, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            Outil expérimental. Ne pas utiliser pour un diagnostic médical réel.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)