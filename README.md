# 📧 SpamShield AI – Email & SMS Spam Detection System

A Machine Learning powered web application that classifies text messages as **Spam or Not Spam** using Natural Language Processing (NLP) and a Multinomial Naive Bayes model.  

This project includes:
- ✅ NLP-based text preprocessing (Tokenization + Stemming)
- ✅ TF-IDF feature extraction
- ✅ Multinomial Naive Bayes classifier
- ✅ Streamlit web interface
- ✅ Real-time prediction with confidence score
- ✅ Lightweight ML deployment on Streamlit Cloud

---

## 🚀 Live Demo

🌐 Live Application: https://spamprediction-ukcchxnzrnmzrfnwegvepm.streamlit.app/  
🔗 GitHub Repository: https://github.com/frhanahmed/SpamPrediction.git  

---

## 🧠 Features

- Enter Email/SMS text manually
- Real-time Spam / Not Spam prediction
- Confidence percentage display
- NLP preprocessing pipeline
- Stopword removal & stemming
- Clean and responsive Streamlit UI
- Sidebar with portfolio & GitHub links
- Integrated contact form

---

## 🏗 System Architecture

Streamlit Frontend  
⬇  
Text Preprocessing (NLTK)  
⬇  
TF-IDF Vectorization  
⬇  
Multinomial Naive Bayes Model  
⬇  
Spam / Not Spam Prediction  

---

## 🛠 Tech Stack

### 🔹 Frontend
- Streamlit
- Python

### 🔹 Machine Learning
- Scikit-learn
- Multinomial Naive Bayes
- TF-IDF Vectorizer
- NumPy
- Pandas (training phase)

### 🔹 NLP
- NLTK
- Tokenization
- Stopword Removal
- Porter Stemmer

### 🔹 Deployment
- Streamlit Community Cloud
- Version Control: Git & GitHub

---

## ⚙️ Production Optimization

During deployment on Streamlit Cloud, the application was optimized to ensure:

- Clean and minimal `requirements.txt`
- Reduced dependency overhead
- Stable Python runtime configuration
- Lightweight ML model instead of deep learning
- Fast startup and low memory usage

These improvements ensured smooth deployment and minimal cold-start delays on the free-tier environment.

---

## 📌 Model Details

- Algorithm: Multinomial Naive Bayes
- Feature Extraction: TF-IDF (max_features = 3000)
- Dataset: SMS Spam Collection Dataset
- Binary Classification:
  - 🚨 Spam
  - ✅ Not Spam
- Text Preprocessing:
  - Lowercasing
  - Tokenization
  - Removal of special characters
  - Stopword filtering
  - Stemming

---

## 📜 Development Workflow

1. Data Cleaning & Preprocessing  
2. Feature Engineering using TF-IDF  
3. Model Training & Evaluation  
4. Pickling Model & Vectorizer  
5. Streamlit Integration  
6. Cloud Deployment  

---

## 👨‍💻 Author

**Farhan Ahmed**  

- LinkedIn: https://www.linkedin.com/in/farhanahmedf21  
- GitHub: https://github.com/frhanahmed  
- Portfolio: https://frhanahmed.github.io/Portfolio/

---

## ⭐ If You Like This Project

Give it a star on GitHub ⭐
