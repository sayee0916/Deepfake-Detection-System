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
# 🔹 Premium Dark AI UI
# -------------------------------

st.set_page_config(
    page_title="Deepfake AI Detector",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}
.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align:center;
    color:#94a3b8;
    margin-bottom:30px;
}
.card {
    background: #1e293b;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 10px 40px rgba(0,0,0,0.4);
}
.real-text {
    font-size: 30px;
    font-weight: 800;
    color: #22c55e;
}
.fake-text {
    font-size: 30px;
    font-weight: 800;
    color: #ef4444;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🧠 Deepfake AI Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Neural Network Powered Image Authenticity System</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

st.write("")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1], gap="large")

    # ---------------- LEFT SIDE ----------------
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- RIGHT SIDE ----------------
    with col2:
        with st.spinner("Running AI Analysis..."):
            img = image.resize(IMG_SIZE)
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            prediction_val = model.predict(img_array, verbose=0)[0][0]

            label = "Real" if prediction_val > 0.5 else "Fake"
            confidence = prediction_val if prediction_val > 0.5 else (1 - prediction_val)

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Strong Result Display
        if label == "Real":
            st.markdown("<div class='real-text'>✅ AUTHENTIC IMAGE</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='fake-text'>⚠ DEEPFAKE DETECTED</div>", unsafe_allow_html=True)

        st.write("")

        st.markdown(f"### Confidence Score: {confidence*100:.2f}%")
        st.progress(float(confidence))

        # ---------------- Donut Chart ----------------
        real_prob = float(prediction_val)
        fake_prob = float(1 - prediction_val)

        fig_donut = go.Figure(data=[go.Pie(
            labels=['Real', 'Fake'],
            values=[real_prob, fake_prob],
            hole=.6,
            marker=dict(colors=["#22c55e", "#ef4444"]),
            textinfo='percent'
        )])

        fig_donut.update_layout(
            paper_bgcolor="#1e293b",
            plot_bgcolor="#1e293b",
            font=dict(color="white"),
            showlegend=True,
            height=300
        )

        st.plotly_chart(fig_donut, use_container_width=True)

        # ---------------- Modern Gauge ----------------
        gauge_color = "#22c55e" if label == "Real" else "#ef4444"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={'font': {'size': 40}},
            title={'text': "Model Confidence (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'bgcolor': "#1e293b",
            }
        ))

        fig.update_layout(
            height=300,
            paper_bgcolor="#1e293b",
            font={'color': "white"}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload an image to begin AI detection.")
