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
# 🔹 Premium Dark Theme
# -------------------------------
st.set_page_config(page_title="Deepfake AI Detector", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #0a0f1c;
    color: white;
}

/* Title */
.main-title {
    font-size: 46px;
    font-weight: 900;
    text-align: center;
    color: white;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align:center;
    color:#cbd5e1;
    margin-bottom:40px;
    font-size:18px;
}

/* Card */
.card {
    background: #111827;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0px 10px 40px rgba(0,0,0,0.6);
}

/* Result Text */
.real-text {
    font-size: 34px;
    font-weight: 900;
    color: #00ff88;
}

.fake-text {
    font-size: 34px;
    font-weight: 900;
    color: #ff3b3b;
}

.confidence-text {
    font-size: 22px;
    font-weight: 700;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🧠 Deepfake AI Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Image Authenticity Verification</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

st.write("")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1], gap="large")

    # ---------------- LEFT ----------------
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- RIGHT ----------------
    with col2:
        with st.spinner("Running AI Model..."):
            img = image.resize(IMG_SIZE)
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            prediction_val = model.predict(img_array, verbose=0)[0][0]

            label = "Real" if prediction_val > 0.5 else "Fake"
            confidence = prediction_val if prediction_val > 0.5 else (1 - prediction_val)

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # RESULT HEADER
        if label == "Real":
            st.markdown("<div class='real-text'>✅ AUTHENTIC IMAGE</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='fake-text'>⚠ DEEPFAKE DETECTED</div>", unsafe_allow_html=True)

        st.write("")

        # CONFIDENCE TEXT
        st.markdown(
            f"<div class='confidence-text'>Confidence Score: {confidence*100:.2f}%</div>",
            unsafe_allow_html=True
        )

        st.progress(float(confidence))

        st.write("")

        # ---------------- DONUT CHART ----------------
        real_prob = float(prediction_val)
        fake_prob = float(1 - prediction_val)

        fig_donut = go.Figure(data=[go.Pie(
            labels=['Real', 'Fake'],
            values=[real_prob, fake_prob],
            hole=.65,
            marker=dict(colors=["#00ff88", "#ff3b3b"]),
            textinfo='percent',
            textfont=dict(size=18, color="white")
        )])

        fig_donut.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="white", size=16),
            showlegend=True,
            height=320,
            legend=dict(font=dict(color="white"))
        )

        st.plotly_chart(fig_donut, use_container_width=True)

        # ---------------- GAUGE ----------------
        gauge_color = "#00ff88" if label == "Real" else "#ff3b3b"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={'font': {'size': 50, 'color': "white"}},
            title={'text': "Model Confidence (%)", 'font': {'size': 20, 'color': "white"}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {'color': gauge_color},
                'bgcolor': "#1f2937",
            }
        ))

        fig.update_layout(
            height=320,
            paper_bgcolor="#111827",
            font={'color': "white"}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload an image to begin AI detection.")
