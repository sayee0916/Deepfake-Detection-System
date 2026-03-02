import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
import plotly.graph_objects as go

# -------------------------------
# 🔹 Configuration
# -------------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "deepfake_model.keras"

@st.cache_resource
def load_deepfake_model():
    base_model = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])

    try:
        model.load_weights(MODEL_PATH)
    except Exception:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)

    return model


model = load_deepfake_model()

# -------------------------------
# 🔹 Professional Clean UI
# -------------------------------

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🧠",
    layout="wide"
)

# Clean modern styling
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}

.header {
    font-size: 40px;
    font-weight: 700;
    color: #111827;
    text-align: center;
    margin-bottom: 5px;
}

.subheader {
    text-align: center;
    color: #6b7280;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.06);
}

.result-real {
    font-size: 30px;
    font-weight: 700;
    color: #16a34a;
}

.result-fake {
    font-size: 30px;
    font-weight: 700;
    color: #dc2626;
}

.conf-text {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>🧠 Deepfake Face Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>AI-powered image authenticity analysis using EfficientNet</div>", unsafe_allow_html=True)

st.write("")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

st.write("")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    # LEFT COLUMN - IMAGE
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT COLUMN - RESULTS
    with col2:
        with st.spinner("Analyzing image..."):
            img = image.resize(IMG_SIZE)
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            prediction_val = model.predict(img_array, verbose=0)[0][0]

            label = "Real" if prediction_val > 0.5 else "Fake"
            confidence = prediction_val if prediction_val > 0.5 else (1 - prediction_val)

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Strong Result Text
        if label == "Real":
            st.markdown("<div class='result-real'>✅ REAL FACE DETECTED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-fake'>⚠ FAKE / DEEPFAKE DETECTED</div>", unsafe_allow_html=True)

        st.write("")

        st.markdown(f"<div class='conf-text'>Confidence: {confidence*100:.2f}%</div>", unsafe_allow_html=True)
        st.progress(float(confidence))

        st.write("")

        # ---------------- Gauge ----------------
        gauge_color = "#16a34a" if label == "Real" else "#dc2626"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Model Confidence (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 50], 'color': "#fee2e2"},
                    {'range': [50, 75], 'color': "#fef3c7"},
                    {'range': [75, 100], 'color': "#dcfce7"},
                ],
            }
        ))

        fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            font={'color': "#111827"}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write("")

        # Breakdown
        st.subheader("Prediction Breakdown")

        real_prob = float(prediction_val)
        fake_prob = float(1 - prediction_val)

        st.bar_chart({
            "Real": [real_prob],
            "Fake": [fake_prob]
        })

        st.metric("Raw Model Score", f"{prediction_val:.4f}")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload an image to begin analysis.")
