import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "final_model.keras",
        custom_objects={"dice_loss": dice_loss, "dice_coefficient": dice_coefficient}
    )

model = load_model()

IMG_SIZE = 256

st.title("Face Segmentation App 🎭")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to image
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Original Image")

    # Preprocess
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Predict
    pred = model.predict(img_input)[0]

    # Convert to binary mask
    pred_mask = (pred > 0.5).astype(np.uint8)

    # Resize back
    pred_mask = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))

    st.image(pred_mask, caption="Predicted Mask", clamp=True)