
import streamlit as st
from PIL import Image
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
project_dir = 'your_path/ImageCaptioning'  # Change to your deployment path
model_path = os.path.join(project_dir, 'model_final.keras')
tokenizer_path = os.path.join(project_dir, 'tokenizer.p')
max_length = 35

# Load model and tokenizer
@st.cache_resource
def load_resources():
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    xception = Xception(include_top=False, pooling='avg')
    return model, tokenizer, xception

model, tokenizer, xception_model = load_resources()

# Feature extraction
def extract_feature_img(image, xception_model):
    image = image.resize((299, 299)).convert('RGB')
    image = np.expand_dims(np.array(image), axis=0)
    image = preprocess_input(image)
    feature = xception_model.predict(image, verbose=0)
    return feature

# Caption generator
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# Streamlit UI
st.title("🖼️ Image Caption Generator")
st.write("Upload an image and get a descriptive caption")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        photo = extract_feature_img(image, xception_model)
        caption = generate_caption(model, tokenizer, photo, max_length)

    st.success("Caption Generated:")
    st.write(f"📝 {caption}")
