# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
import gdown

# Set page config - should be the first Streamlit command
st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="wide"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_resources()

# Load your saved assets from Google Drive
@st.cache_resource
def load_assets():
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Google Drive file IDs (replace these with your actual file IDs)
        # To get the file ID from your Google Drive path:
        # /content/drive/MyDrive/Mental-Health-Classification/models/
        
        # You need to get the shareable links for these files first:
        model_file_id = "1hpM7nicGeetKdTRO564pwm1-yP6xvsTw"
        vectorizer_file_id = "1r0KNQRZBNWEYhauJB-rNKiO_W-nnVWux"
        class_names_file_id = "1hzUflkNWjiOzSH65kysz3uxBB0uhsbpW"
        
        # Download files from Google Drive
        model_url = f"https://drive.google.com/uc?id={model_file_id}"
        vectorizer_url = f"https://drive.google.com/uc?id={vectorizer_file_id}"
        class_names_url = f"https://drive.google.com/uc?id={class_names_file_id}"
        
        # Download files if they don't exist locally
        model_path = "models/mental_health_predictor.joblib"
        vectorizer_path = "models/text_vectorizer.joblib"
        class_names_path = "models/class_names.joblib"
        
        if not os.path.exists(model_path):
            gdown.download(model_url, model_path, quiet=False)
            st.success("✅ Model file downloaded successfully")
        if not os.path.exists(vectorizer_path):
            gdown.download(vectorizer_url, vectorizer_path, quiet=False)
            st.success("✅ Vectorizer file downloaded successfully")
        if not os.path.exists(class_names_path):
            gdown.download(class_names_url, class_names_path, quiet=False)
            st.success("✅ Class names file downloaded successfully")
        
        # Load the downloaded files
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        class_names = joblib.load(class_names_path)
        
        st.success("✅ All model files loaded successfully!")
        return model, vectorizer, class_names
        
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

# Load assets
model, vectorizer, class_names = load_assets()

# Define the text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# App UI
st.title("🧠 Mental Health Statement Classifier")
st.markdown("This app predicts the mental health category based on a personal statement. Enter text below and click **Predict**.")

if model is None:
    st.error("❌ Model files could not be loaded. Please check the Google Drive links.")
    st.info("ℹ️ To get the file IDs:")
    st.write("1. Go to your Google Drive")
    st.write("2. Right-click on each model file → 'Get link'")
    st.write("3. Set sharing to 'Anyone with the link can view'")
    st.write("4. Copy the file ID from the URL (the part after '/d/' and before '/view')")
    st.write("5. Replace the placeholders in the code with your actual file IDs")
else:
    user_input = st.text_area(
        "**Enter a statement:**",
        height=150,
        placeholder="e.g., I've been feeling incredibly restless and worried for the past month, can't sleep properly..."
    )

    predict_button = st.button("🚀 Predict", type="primary")

    if predict_button:
        if user_input.strip():
            with st.spinner('🧠 Analyzing statement...'):
                cleaned_input = clean_text(user_input)
                transformed_input = vectorizer.transform([cleaned_input])
                prediction = model.predict(transformed_input)
                probabilities = model.predict_proba(transformed_input)[0]

            st.success("### Prediction Results")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric(label="**Predicted Category**", value=prediction[0])
                predicted_index = np.where(class_names == prediction[0])[0][0]
                confidence = probabilities[predicted_index]
                st.metric(label="**Confidence**", value=f"{confidence:.2%}")

            with col2:
                st.subheader("Confidence Breakdown")
                prob_data = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
                st.bar_chart(prob_data)

            with st.expander("See how the text was processed"):
                st.code(cleaned_input)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

st.markdown("---")
st.markdown("""
**📖 How it works:**
1. You type a personal statement into the box.
2. The app cleans and processes the text.
3. It converts the text into numerical features using TF-IDF.
4. The trained Random Forest model makes a prediction.
5. Results are displayed along with the model's confidence.

**🔒 Privacy Note:** This app runs entirely in your browser. No data is sent to any server.

**🎓 Purpose:** This is a prototype for educational purposes.
""")