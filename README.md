# 🧠 Mental Health Statement Classifier

A Streamlit web application that classifies mental health conditions based on text input using machine learning and Natural Language Processing (NLP).

![Mental Health Classifier](https://img.shields.io/badge/Streamlit-1.28.1-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-0078D7)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-FF6F00)

## 🌟 Live Demo

Check out the live application:  
👉 [Mental Health Classifier App](https://mental-health-classifier-ewtdikafcatpwha3juzkof.streamlit.app/)

## 📋 Overview

This application uses a trained machine learning model to analyze personal statements and classify them into mental health categories:

- **Normal**
- **Depression**
- **Suicidal**
- **Anxiety**
- **Bipolar**
- **Stress**
- **Personality Disorder**

The app processes text using NLP techniques including tokenization, stopword removal, and lemmatization, then uses a TF-IDF vectorizer and Logistic Regression classifier for prediction.

## 🚀 Features

- **Text Analysis**: Clean and process user input using advanced NLP techniques
- **Real-time Prediction**: Instant classification with confidence scores
- **Visual Results**: Interactive bar charts showing prediction probabilities
- **Privacy-Focused**: All processing happens locally in your browser
- **Demo Mode**: Fallback functionality when model files aren't available

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn, Joblib
- **NLP**: NLTK, TF-IDF Vectorization
- **Data Processing**: NumPy, Pandas
- **File Management**: Gdown (for Google Drive integration)

## 📁 Project Structure

```
Mental-Health-Classifier/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
└── models/                 # Model files
├── best_deployment_model.joblib
├── tfidf_vectorizer.joblib
└── class_names.joblib''
```

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KhinChaw/Mental-Health-Classifier.git
   cd Mental-Health-Classifier
   ```
