import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("PROJECT 10 IMAGE CLASSIFIER/binary_image_classifier.keras")

# Class labels
classes = ["Female", "Male"]

st.set_page_config(
    page_title="Gender Classification",
    page_icon="🧑",
    layout="centered"
)

st.title("🧑 Gender Classification App")
st.write("Upload an image to predict whether it is Male or Female.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        label = classes[1]
        confidence = prediction[0][0]
    else:
        label = classes[0]
        confidence = 1-prediction[0][0]

    st.success(f"Prediction : {label}")
    st.info(f"Confidence : {confidence*100:.2f}%")
