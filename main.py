from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse,JSONResponse
from keras.applications import DenseNet201
from keras.utils import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

from keras.models import Model
from PIL import Image
import numpy as np

# Load the DenseNet201 model once to avoid reloading it for every request
base_model = DenseNet201()
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)


with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
max_length = 34

caption_model = load_model("models/caption_model.keras")

app = FastAPI()

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
async def predict(file: UploadFile = File(...)):
    """
    API endpoint to extract features from an uploaded image file.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        return {"error": "Only JPEG or PNG images are supported."}
    try:
        # File metadata
        filename = file.filename
        img_size = 224

        # Open the uploaded image file
        img = Image.open(file.file)  # 'file.file' is a SpooledTemporaryFile
        
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((img_size, img_size))  # Resize the image

        # Extract features
        feature = extract_features(img)

        caption = predict_caption(caption_model, tokenizer, max_length, feature)
        
        return JSONResponse({
            "filename": filename,
            "caption": caption,
        })
            # "features": feature.tolist()  # Convert NumPy array to list for JSON serialization

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

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
    return caption
