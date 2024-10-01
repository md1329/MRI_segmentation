import cv2
import numpy as np

# Preprocess the MRI image before passing to the model
def preprocess_image(image):
    # Convert the image to grayscale (assuming MRI is grayscale)
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the input size expected by the model (e.g., 256x256)
    resized_image = cv2.resize(gray_image, (256, 256))
    
    # Normalize the image (range between 0 and 1)
    normalized_image = resized_image / 255.0
    
    # Add batch dimension (1, 256, 256, 1) for model input
    return np.expand_dims(np.expand_dims(normalized_image, axis=-1), axis=0)

# Postprocess the prediction output to generate a binary mask
def postprocess_prediction(prediction):
    # Assuming the model outputs a probability map, apply a threshold
    mask = (prediction > 0.5).astype(np.uint8) * 255
    return mask[0, :, :, 0]  # Return the mask for display
