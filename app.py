import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Caching the model to avoid reloading it on every interaction
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('CNN_fruits.h5')
    return model

model = load_model()

classes = ['Banana', 'Orange', 'Apple']

st.title('Fruit Classification Application')
st.write("Upload an image of a fruit to identify if it's a **banana**, **orange**, or **apple**.")

uploaded_file = st.file_uploader("Upload an Image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # 'use_column_width' is deprecated; use_container_width is the current parameter
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((32, 32))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0) # Added verbose=0 to reduce console clutter
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.subheader("Prediction Results")
    st.success(f"**Predicted Fruit:** {predicted_class}")
    st.write(f"Confidence Score: **{confidence:.2f}%**")

    st.write("### Class Predictions")
    # Displaying probabilities for all classes
    for i, label in enumerate(classes):
        st.write(f"**{label}:** {predictions[0][i]*100:.2f}%")
        st.progress(float(predictions[0][i]))
else:
    st.info("Please upload an image to classify.")