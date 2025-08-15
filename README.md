# Next Word Prediction Using LSTM

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-brightgreen)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Project Overview
This project develops a **Next Word Prediction model** using **LSTM networks**.  
Given a sequence of words, the model predicts the most probable next word. We use Shakespeare's *Hamlet* as the dataset for training, providing rich and complex language patterns.

**Key Features:**
- Real-time next-word predictions via a Streamlit web app.
- LSTM-based deep learning architecture.
- Early stopping to prevent overfitting.

---

## 🗂 Dataset
- **Source:** Shakespeare's *Hamlet*
- **Type:** Text corpus
- **Preprocessing:** 
  - Tokenization
  - Sequence generation
  - Padding to ensure uniform input length

---

## 🏗 Model Architecture
- **Embedding Layer:** Converts words to dense vectors.
- **Two LSTM Layers:** For sequence learning and context capture.
- **Dense Output Layer:** Softmax activation to predict probabilities of next word.

---

## 💻 Installation
1. Clone the repository:
```bash
git clone https://github.com/vardhanthadala/next-word-prediction.git
cd next-word-prediction

2. Install dependencies
pip install -r requirements.txt


3. Run the Streamlit app
streamlit run app.py
