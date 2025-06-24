#streamlit file

import streamlit as st
import gdown
from PIL import Image
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_ID = "1-fYifJQREz1qLdksobo2vo7Uvt3Z6cZQ"
TOKENIZER_ID = "1V1eoR_19JD0XXlXRtVjy9bkVmYrXdTaM"

# Local filenames to save downloaded files
model_path = "model_final.keras"
tokenizer_path = "tokenizer.p"
max_length = 35

# Download model and tokenizer from Google Drive if not already present
if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", model_path, quiet=False)

if not os.path.exists(tokenizer_path):
    gdown.download(f"https://drive.google.com/uc?id={TOKENIZER_ID}", tokenizer_path, quiet=False)
    
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
    
st.set_page_config(page_title="Image Caption Generator", layout="centered")

# Streamlit UI
st.title(" Image Caption Generator")
st.write("Upload an image and get a descriptive caption")

uploaded_file1 = st.file_uploader("Choose first image...", type=["jpg", "jpeg", "png"], key="uploader1")
uploaded_file2 = st.file_uploader("Choose second image...", type=["jpg", "jpeg", "png"], key="uploader2")

if uploaded_file1:
    st.image(uploaded_file1, caption="First Image", use_column_width=True)
    # Call your captioning model here for uploaded_file1

if uploaded_file2:
    st.image(uploaded_file2, caption="Second Image", use_column_width=True)
    
    with st.spinner("Generating caption..."):
        photo = extract_feature_img(image, xception_model)
        caption = generate_caption(model, tokenizer, photo, max_length)

    st.success("Caption Generated:")
    st.write(f" {caption}")
