from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
from utils import preprocess_image, postprocess_prediction

# Initialize FastAPI app
app = FastAPI()

# Load your trained model
model = load_model('models/best_model.h5')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    image = Image.open(BytesIO(await file.read()))
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make a prediction using the loaded model
    prediction = model.predict(preprocessed_image)
    
    # Postprocess the output to generate the segmentation mask
    mask = postprocess_prediction(prediction)
    
    # Encode the mask as a PNG image to send back
    _, im_png = cv2.imencode(".png", mask)
    
    # Return the mask as bytes in the response
    return {"mask": im_png.tobytes()}
