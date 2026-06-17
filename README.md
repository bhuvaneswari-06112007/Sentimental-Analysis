 AI-Based Email Classification System

An end-to-end, AI-powered Natural Language Processing (NLP) and Machine Learning pipeline that automatically categorizes incoming emails into predefined organizational departments (e.g., IT Support, Finance, HR) and detects urgency levels to optimize routing and response workflows.

---

📌 Project Overview
Modern organizations process massive volumes of internal and external emails daily. Manual sorting is labor-intensive, slow, and prone to routing errors. This project automates the workflow by analyzing email subjects and bodies using advanced text preprocessing and machine learning classification algorithms to predict destination categories and priority scores instantly.

 Key Features
* **Dual-Field Analysis:** Combines Email Subject and Body context for prediction.
* **Robust NLP Pipeline:** Custom text cleansing, tokenization, stop-word removal, and TF-IDF vectorization.
* **Multi-Model Comparison:** Evaluates Naive Bayes, Logistic Regression, and Random Forest architectures.
* **Priority Detection:** Evaluates email urgency (Low, Medium, High) alongside departmental categorization.
* **Interactive Dashboard:** Built with Streamlit for real-time inference, confidence scoring, and local prediction tracking.

---

 📐 System Architecture

```text
Email Dataset (Kaggle) ➔ Text Preprocessing ➔ Feature Extraction (TF-IDF)
                                                       │
                                                       ▼
Streamlit Dashboard    ◀─── Prediction Engine ◀─── Trained ML Model

📂 Project Phases & Directory Structure

├── data/                  # Raw and processed datasets
├── notebooks/             # EDA and experimental model training
├── src/
│   ├── preprocessing.py   # Text cleansing pipeline
│   ├── train.py           # Model training and comparison scripts
│   └── app.py             # Streamlit Dashboard implementation
├── models/                # Saved serialized models (.pkl)
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
