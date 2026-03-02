import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

# -------------------------------
# 🔹 Configuration
# -------------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "deepfake_model.keras"

@st.cache_resource
def load_deepfake_model():
    # 1. Build the base model
    base_model = EfficientNetB0(
        weights=None, 
        include_top=False, 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # 2. Rebuild architecture based on your error logs:
    # The error (1280, 1) means it goes GAP -> Dense(1) 
    # There is NO Dense(128) layer in your saved file.
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid') # Changed from 128 to 1 to match your file
    ])

    # 3. Load weights
    try:
        # We use load_weights because load_model has a bug with EfficientNet in Keras 3
        model.load_weights(MODEL_PATH)
    except Exception as e:
        st.error(f"Architecture Mismatch: {e}")
        st.info("Attempting fallback: loading full model directly...")
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
        
    return model

# Initialize Model
model = load_deepfake_model()

# -------------------------------
# 🔹 Streamlit Interface
# -------------------------------
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🧠 Deepfake Face Detector")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        # Preprocess
        img = image.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        # We use [0][0] because prediction comes back as a 2D array [[value]]
        prediction_val = model.predict(img_array, verbose=0)[0][0]
        
        # Display Result
        # Adjust threshold logic if needed (is 1 Real or 0 Real?)
        label = "Real" if prediction_val > 0.5 else "Fake"
        confidence = prediction_val if prediction_val > 0.5 else (1 - prediction_val)

        st.divider()
        color = "green" if label == "Real" else "red"
        st.subheader(f"Result: :{color}[{label}]")
        st.write(f"**Confidence Score:** {confidence*100:.2f}%")
        st.progress(float(confidence))

else:
    st.info("Waiting for image upload...")