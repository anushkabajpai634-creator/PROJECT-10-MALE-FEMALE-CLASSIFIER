import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_covid_model():
    return load_model("model.keras")  # or model.h5 if using H5

model = load_covid_model()

IMG_SIZE = (299, 299)

# -----------------------------
# Prediction Function
# -----------------------------
def predict(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    probability = float(prediction[0][0])

    if probability > 0.5:
        label = "COVID-19"
        confidence = probability
    else:
        label = "NORMAL"
        confidence = 1 - probability

    return label, confidence

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 COVID-19 Detection from Chest X-ray")
st.write("Upload a Chest X-ray image to predict whether it is **COVID-19** or **Normal**.")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        with st.spinner("Analyzing..."):
            label, confidence = predict(img)

        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

        if label == "COVID-19":
            st.error("⚠️ COVID-19 detected.")
        else:
            st.success("✅ Normal Chest X-ray.")
