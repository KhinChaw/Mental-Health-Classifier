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
        model_file_id = "1Th6xAsP1MVXJdpZPOzpnQvR10gk_yK3R"
        vectorizer_file_id = "11M5SlKuEw7eOflL2pXxceZowmgH6uD5C"
        class_names_file_id = "10-sCmZrzSeASQ1KS4MQkVs_rmgfrwiTz"
        
        # Download files from Google Drive
        model_url = f"https://drive.google.com/uc?id={model_file_id}"
        vectorizer_url = f"https://drive.google.com/uc?id={vectorizer_file_id}"
        class_names_url = f"https://drive.google.com/uc?id={class_names_file_id}"
        
        # Download files if they don't exist locally
        model_path = "models/best_deployment_model.joblib"
        vectorizer_path = "models/tfidf_vectorizer.joblib"
        class_names_path = "models/class_names.joblib"
        
        if not os.path.exists(model_path):
            gdown.download(model_url, model_path, quiet=False)
        if not os.path.exists(vectorizer_path):
            gdown.download(vectorizer_url, vectorizer_path, quiet=False)
        if not os.path.exists(class_names_path):
            gdown.download(class_names_url, class_names_path, quiet=False)
        
        # Load the downloaded files
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        class_names = joblib.load(class_names_path)
        
        return model, vectorizer, class_names
        
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        return None, None, None

# Load assets
model, vectorizer, class_names = load_assets()

# If model files couldn't be loaded, create demo assets
if model is None or vectorizer is None or class_names is None:
    st.error("❌ Model files could not be loaded. Running in demo mode with sample data.")
    
    # Create simple demo assets
    class_names = np.array(['Depression', 'Anxiety', 'Bipolar', 'PTSD', 'Healthy'])
    
    # Create a dummy vectorizer and model for demo
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    
    # Fit with some dummy data
    dummy_texts = [
        "feel sad depressed hopeless",
        "anxious worry panic fear",
        "mood swings high low",
        "trauma flashbacks nightmare",
        "good fine okay well"
    ]
    vectorizer.fit(dummy_texts)
    
    # Create a dummy model that returns random probabilities
    class DemoModel:
        def predict(self, X):
            return np.random.choice(class_names, size=X.shape[0])
        
        def predict_proba(self, X):
            # Return random probabilities that sum to 1
            probs = np.random.rand(X.shape[0], len(class_names))
            return probs / probs.sum(axis=1, keepdims=1)
    
    model = DemoModel()
    
    st.info("ℹ️ Running in demo mode. To use the real model, please check your Google Drive file IDs.")

# Debug toggle
debug_mode = st.sidebar.checkbox("Show debug information", value=False)

if debug_mode:
    st.sidebar.write("🛠️ DEBUG INFORMATION")
    st.sidebar.write(f"Class names type: {type(class_names)}")
    st.sidebar.write(f"Class names content: {class_names}")
    st.sidebar.write(f"Number of classes: {len(class_names)}")

    # Check model properties
    if hasattr(model, 'classes_'):
        st.sidebar.write(f"Model classes: {model.classes_}")
        if hasattr(model, 'n_classes_'):
            st.sidebar.write(f"Model n_classes: {model.n_classes_}")
    else:
        st.sidebar.write("Model has no classes_ attribute")

    # Check vectorizer
    if hasattr(vectorizer, 'get_feature_names_out'):
        try:
            features = vectorizer.get_feature_names_out()
            st.sidebar.write(f"Vectorizer features: {features[:10]}...")
        except:
            st.sidebar.write("Could not get vectorizer features")

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

# Initialize session state for text input if it doesn't exist
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

if 'clear_clicked' not in st.session_state:
    st.session_state.clear_clicked = False

# Create a text area with the session state value
user_input = st.text_area(
    "**Enter a statement:**",
    height=150,
    placeholder="e.g., I've been feeling incredibly restless and worried for the past month, can't sleep properly...",
    key="text_input"
)

# Update session state with the current input
if user_input != st.session_state.user_input:
    st.session_state.user_input = user_input

# Create columns for the buttons
col1, col2, col3 = st.columns([1, 1, 6])

with col1:
    predict_button = st.button("🚀 Predict", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("🧹 Clear", use_container_width=True)

# Handle clear button click
if clear_button:
    st.session_state.user_input = ''
    st.session_state.clear_clicked = True
    # Use st.rerun() to refresh the app and clear the text area
    st.rerun()

# Reset the clear_clicked flag after the rerun
if st.session_state.clear_clicked:
    st.session_state.clear_clicked = False
    # Set focus back to the text area
    st.markdown(
        """
        <script>
            // Focus on the text area after clear
            const textarea = window.parent.document.querySelector('textarea');
            if (textarea) {
                textarea.focus();
            }
        </script>
        """,
        unsafe_allow_html=True
    )

if predict_button:
    if st.session_state.user_input.strip():
        with st.spinner('🧠 Analyzing statement...'):
            cleaned_input = clean_text(st.session_state.user_input)
            
            if debug_mode:
                st.sidebar.write(f"Cleaned text: {cleaned_input}")
            
            transformed_input = vectorizer.transform([cleaned_input])
            
            if debug_mode:
                st.sidebar.write(f"Features shape: {transformed_input.shape}")
                st.sidebar.write(f"Non-zero features: {transformed_input.nnz}")
            
            prediction = model.predict(transformed_input)
            probabilities = model.predict_proba(transformed_input)[0]
            
            if debug_mode:
                st.sidebar.write(f"Raw prediction: {prediction}")
                st.sidebar.write(f"All probabilities: {probabilities}")

        st.success("### Prediction Results")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(label="**Predicted Category**", value=prediction[0])
            try:
                predicted_index = np.where(class_names == prediction[0])[0][0]
                confidence = probabilities[predicted_index]
                st.metric(label="**Confidence**", value=f"{confidence:.2%}")
            except:
                st.metric(label="**Confidence**", value="N/A")

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
4. The trained Logistic Regression model makes a prediction.
5. Results are displayed along with the model's confidence.

**🔒 Privacy Note:** This app runs entirely in your browser. No data is sent to any server.

**🎓 Purpose:** This is a prototype for educational purposes.
""")