import streamlit as st
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from keras.applications import DenseNet201
from keras.utils import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from keras.models import Model
from PIL import Image
import numpy as np
from io import BytesIO
import requests

# Load pre-trained models and tokenizer
@st.cache_resource
def load_models():
    base_model = DenseNet201(weights=None)
    base_model.load_weights('models/densenet201_weights_tf_dim_ordering_tf_kernels.h5')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    caption_model = load_model("models/caption_model.keras")
    return feature_extractor, tokenizer, caption_model

feature_extractor, tokenizer, caption_model = load_models()
max_length = 34

# FastAPI app setup
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper Functions
def extract_features(img):
    img = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    feature = feature_extractor.predict(img, verbose=0)
    return feature[0]

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, tokenizer, max_length, feature):
    feature = np.expand_dims(feature, axis=0)
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return " ".join(in_text.split(" ")[1:-1])  # Remove "startseq" and "endseq"

@fastapi_app.post("/predict/url")
async def predict_from_url(image_url: str):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((224, 224))
        feature = extract_features(img)
        caption = predict_caption(caption_model, tokenizer, max_length, feature)
        return {"url": image_url, "caption": caption}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@fastapi_app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((224, 224))
        feature = extract_features(img)
        caption = predict_caption(caption_model, tokenizer, max_length, feature)
        return {"filename": file.filename, "caption": caption}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# Streamlit + FastAPI Integration
st.title("Image Captioning Service with Streamlit and FastAPI")
st.write("""
This app allows users to generate captions for images via API endpoints exposed by FastAPI. 
Use `/predict/url` or `/predict/file` for captioning.
""")

st.write("### Test API Endpoints")

# Test URL Endpoint
image_url = st.text_input("Enter Image URL:")
if st.button("Generate Caption from URL"):
    if image_url:
        try:
            response = requests.post(
                "/predict/url",
                json={"image_url": image_url}
            )
            if response.status_code == 200:
                st.success(f"Caption: {response.json()['caption']}")
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

# Test File Upload Endpoint
uploaded_file = st.file_uploader("Upload an Image File", type=["jpg", "jpeg", "png"])
if st.button("Generate Caption from File") and uploaded_file:
    try:
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(
            "/predict/file",
            files=files
        )
        if response.status_code == 200:
            st.success(f"Caption: {response.json()['caption']}")
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")

# Mount FastAPI app to Streamlit
st.write("### FastAPI Endpoints")
st.write("You can also test endpoints directly via tools like Postman.")

fastapi_app.add_middleware(WSGIMiddleware)
