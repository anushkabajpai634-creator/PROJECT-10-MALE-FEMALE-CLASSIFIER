import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Eye Detection", page_icon="👁", layout="centered")

st.title("👁 Eye Detection App")
st.write("Upload an eye image to get the prediction.")

# ----------------------------
# Load TFLite Model
# ----------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="PROJECT 12 EYE CLASSIFIER/my_eye_detection_model_quantized.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Get expected input size
    input_shape = input_details[0]["shape"]

    height = input_shape[1]
    width = input_shape[2]

    img = image.resize((width, height))
    img = np.array(img)

    # Normalize if required
    if input_details[0]["dtype"] == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(input_details[0]["dtype"])

    img = np.expand_dims(img, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])

    st.subheader("Prediction")

    # Binary classification
    if output.shape[-1] == 1:
        probability = float(output[0][0])

        if probability > 0.5:
            prediction = "Open Eye"
        else:
            prediction = "Closed Eye"

        st.success(prediction)
        st.write(f"Confidence: {probability:.2f}")

    # Multi-class classification
    else:
        labels = [
            "Closed Eye",
            "Open Eye"
        ]

        predicted_index = np.argmax(output)
        confidence = np.max(output)

        if predicted_index < len(labels):
            prediction = labels[predicted_index]
        else:
            prediction = f"Class {predicted_index}"

        st.success(prediction)
        st.write(f"Confidence: {confidence:.2f}")
