# Next Word Prediction Using LSTM

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-brightgreen)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Project Overview
This project develops a **Next Word Prediction model** using **LSTM networks**.  
Given a sequence of words, the model predicts the most probable next word. We use Shakespeare's *Hamlet* as the dataset for training, providing rich and complex language patterns.

**Try it live:** [Open Streamlit App](https://lstmrnnnextwordprediction-cdxw7yuaspjfzuf6cwzy2m.streamlit.app/)

## 🛠 Tech Stack
- **Python 3.10**
- **TensorFlow 2.x** (for LSTM model)
- **Keras** (for deep learning layers)
- **Streamlit** (for web app deployment)
- **NumPy & Pandas** (for data processing)
- **NLTK / Tokenizer** (for text preprocessing)

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
```
## 2. Install dependencies
```bash
pip install -r requirements.txt

```
## 3. Run the Streamlit app
```bash
streamlit run app.py

```
## 🎯 Usage
```bash
Open the Streamlit web app in your browser.

Enter a sequence of words in the input box.

Click Predict to get the next word.

Example:

Input: "To be, or not to"
Predicted next word: "be"
```
## 🧠 Model Training
```bash
Loss Function: Categorical Cross-Entropy

Optimizer: Adam

Early Stopping: Monitors validation loss to avoid overfitting

Training/Validation Split: 80/20

Training Code Example:

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping]
)
```
## 📈 Evaluation
```bash
The model is tested on example sentences to evaluate next-word prediction accuracy.

Performs well on common patterns.

Rare or complex phrases may yield less accurate predictions.
```
## 🔮 Future Work
```bash
Expand dataset for better generalization.

Experiment with Transformer-based models for improved accuracy.

Deploy as a REST API for wider usage.
```
## 🤝 Contributing
```bash
Contributions are welcome! Open an issue or submit a pull request for improvements or bug fixes.
```
