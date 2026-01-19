import streamlit as st
import joblib  # Used for Scikit-Learn models
import numpy as np
from PIL import Image, ImageOps

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Product Pricing", page_icon="💰")

# --- MODEL LOADING ---
@st.cache_resource
def load_my_model():
    try:
        # Load using joblib because your .h5 file contains a Scikit-Learn model
        model = joblib.load('product_price_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# --- PREPROCESSING LOGIC ---
def preprocess_image(image):
    # 1. Resize image to match the size used during training
    size = (224, 224) 
    image = ImageOps.fit(image, size, Image.LANCZOS)
    
    # 2. Convert to numpy array and normalize
    img_array = np.asarray(image).astype(np.float32) / 255.0

    # 3. IMPORTANT: Linear Regression expects a flat 1D array of features
    # Your model has 150,528 features (224*224*3 pixels)
    # Change this:


# To this (just for testing):
    flattened_features = img_array.flatten()[:129].reshape(1, -1)
    return flattened_features

# --- DIALOG FOR PREDICTION ---
@st.dialog("Predict Product Price")
def prediction_dialog():
    st.write("Upload an image of the product to estimate its market value.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Product', use_container_width=True)

        # --- Replace this section in your app.py ---
        if st.button("Predict Cost"):
            if model is not None:
                with st.spinner('Calculating price...'):
            # 1. Preprocess
                    processed_data = preprocess_image(image)
            
            # 2. Predict
                    prediction = model.predict(processed_data)

            # 3. FIX: Flatten the output and take the first element
            # This handles [[value]] or [value] formats
                    predicted_price = float(np.array(prediction).flatten()[0])

                    st.success(f"### Estimated Price: ${predicted_price:.2f}")

# --- MAIN UI ---
def main():
    st.title("💰 Smart Product Cost Predictor")
    st.markdown("""
    This app uses a **Machine Learning (Linear Regression)** model to analyze 
    image pixels and suggest a competitive market price.
    """)

    if st.button("Launch Price Predictor"):
        prediction_dialog()

if __name__ == "__main__":
    main()