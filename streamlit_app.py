"""
=========================================================
FACIAL EMOTION RECOGNITION — STREAMLIT WEB APP (L5/L6)
Author: CoreyLeath-code (Corey Leath)
Description:
    - Provides a clean web UI for emotion prediction
    - Allows image uploads
    - Displays confidence scores
=========================================================
"""

import io

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import EmotionCNN, EMOTION_LABELS

MODEL_PATH = "models/emotion_cnn.pth"


# ---------------------------------------------------------
# Load model once (performance optimized)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = EmotionCNN()
    try:
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
# Preprocessing transform
# ---------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


model, load_error = load_model()

# ---------------------------------------------------------
# Page Settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🙂",
    layout="centered"
)

st.title("🎭 Facial Emotion Recognition System")
st.write("Upload an image and let the model detect the emotion.")

if load_error:
    st.warning(
        f"⚠️ Model weights not found at `{MODEL_PATH}`. "
        "Please train the model first and ensure `MODEL_PATH` points to the saved weights."
    )

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is None:
        st.error("Cannot run inference — model weights are not loaded.")
    else:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

        # Show uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        input_tensor = preprocess(image).unsqueeze(0)  # (1, 1, 48, 48)

        # Prediction
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        emotion = EMOTION_LABELS[pred_idx]
        confidence = float(probs[pred_idx])

        # -----------------------------------------------------
        # Display result
        # -----------------------------------------------------
        st.subheader("🎯 Prediction Result")
        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Confidence:** {confidence:.4f}")

        # Confidence bar chart
        st.subheader("📊 Confidence Scores")
        confidence_dict = {EMOTION_LABELS[i]: float(probs[i]) for i in range(7)}
        st.bar_chart(confidence_dict)

else:
    st.info("Please upload an image to get started.")
