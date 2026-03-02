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
# 🔹 Advanced UI Styling
# -------------------------------

st.set_page_config(
    page_title="Deepfake AI Detector",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}
.big-title {
    font-size: 50px;
    font-weight: 800;
    text-align: center;
    background: -webkit-linear-gradient(#00C6FF, #0072FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align:center;
    font-size:18px;
    opacity:0.8;
    margin-bottom:30px;
}
.glass-card {
    padding:25px;
    border-radius:20px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.15);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🧠 Deepfake AI Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced Neural Network powered authenticity detection</div>", unsafe_allow_html=True)

st.divider()

uploaded_file = st.file_uploader("📤 Upload Face Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    # ---------------- LEFT SIDE ----------------
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- RIGHT SIDE ----------------
    with col2:
        with st.spinner("🔍 Running AI Analysis..."):
            img = image.resize(IMG_SIZE)
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            prediction_val = model.predict(img_array, verbose=0)[0][0]

            label = "Real" if prediction_val > 0.5 else "Fake"
            confidence = prediction_val if prediction_val > 0.5 else (1 - prediction_val)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        # Result Badge
        if label == "Real":
            st.success("✅ Authentic Face Detected")
        else:
            st.error("⚠️ Deepfake Detected")

        st.markdown(f"### 🎯 Confidence: {confidence*100:.2f}%")
        st.progress(float(confidence))

        # ---------------- Gauge Chart ----------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Model Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "lime" if label == "Real" else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "#440000"},
                    {'range': [50, 75], 'color': "#555500"},
                    {'range': [75, 100], 'color': "#004400"},
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- Probability Breakdown ----------------
        st.markdown("### 📊 Probability Breakdown")

        real_prob = prediction_val
        fake_prob = 1 - prediction_val

        st.bar_chart({
            "Real": [float(real_prob)],
            "Fake": [float(fake_prob)]
        })

        # Raw Score
        st.metric("Raw Model Score", f"{prediction_val:.4f}")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👆 Upload an image to start AI detection")

st.divider()
st.caption("⚡ Powered by EfficientNetB0 • Built with TensorFlow & Streamlit")
