from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse,JSONResponse
from keras.applications import DenseNet201
from keras.utils import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from fastapi.middleware.cors import CORSMiddleware
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses TensorFlow info and warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppresses TensorFlow info and warnings

from keras.models import Model
from PIL import Image
import numpy as np

from pydantic import BaseModel, HttpUrl
import requests
from io import BytesIO

# Load the DenseNet201 model once to avoid reloading it for every request
base_model = DenseNet201()
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)


with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
max_length = 34

caption_model = load_model("models/caption_model.keras")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend URL, e.g., ["https://your-website-a.com"]
    allow_credentials=True,
    allow_methods=["*"],  # or specify allowed HTTP methods, e.g., ["GET", "POST"]
    allow_headers=["*"],  # or specify allowed headers
)

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/upload", response_class=HTMLResponse)
async def render_form():
    """Serve the HTML form to upload an image."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Image</title>
    </head>
    <body>
        <h2>Upload an Image</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    """


@app.post("/predict")
async def predict(request: Request):
    # Parse JSON body from the incoming request
    request_body = await request.json()
    print("Received request body:", request_body)
    url = request_body.get("url")       # Extract the 'url' from the request body
    print("Received URL:", url)

    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Open the image using PIL
    img = Image.open(BytesIO(response.content))

    # Convert to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize the image
    img_size = 224
    img = img.resize((img_size, img_size))

    # Extract features and generate caption (your existing logic)
    feature = extract_features(img)
    caption = predict_caption(caption_model, tokenizer, max_length, feature)
    print("url : ", url, "caption", caption)

    # Return the result
    return JSONResponse({
        "url": url,
        "caption": caption,
    })

def extract_features(img):
    """
    Extract features from a given image using the pre-trained DenseNet201 model.
    """
    # Convert the image to a NumPy array
    img = img_to_array(img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Use the feature extractor model to predict features
    feature = feature_extractor.predict(img, verbose=0)

    return feature[0]  # Return the feature vector

def idx_to_word(integer,tokenizer):

    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def predict_caption(model, tokenizer, max_length, feature):

    feature = np.expand_dims(feature, axis=0)
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([feature,sequence])
        y_pred = np.argmax(y_pred)

        print("Feature shape:", feature.shape)
        print("Sequence shape:", sequence.shape)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text+= " " + word

        if word == 'endseq':
            break
    
    # Remove "startseq" and "endseq" from the caption
    caption = in_text.split(" ")[1:-1]  # Split into words and remove the first and last tokens
    return " ".join(caption)  # Rejoin the remaining words into a single string
