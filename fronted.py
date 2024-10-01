import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Streamlit UI
st.title("Brain MRI Metastasis Segmentation")
st.write("Upload a brain MRI image and get the metastasis segmentation result.")

# Upload MRI image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("Segment"):
        # Convert the image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Send the image to FAST API backend for prediction
        response = requests.post("http://127.0.0.1:8000/predict/", files={"file": img_bytes})

        if response.status_code == 200:
            # Get the segmentation mask from the response
            mask_data = BytesIO(response.json()["mask"])
            mask_image = Image.open(mask_data)

            # Display the segmentation mask
            st.image(mask_image, caption="Metastasis Segmentation", use_column_width=True)
        else:
            st.error("Error in prediction request")
