import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.preprocessing import preprocess_df, fit_tfidf

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the model on test data and returns metrics and confusion matrix.
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n==================== {model_name} Evaluation ====================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'model': model
    }

def train_pipeline():
    # 1. Load Data
    data_path = 'data/synthetic_emails.csv'
    if not os.path.exists(data_path):
        print(f"[INFO] '{data_path}' not found. Generating synthetic data first...")
        from generate_data import generate_synthetic_data
        generate_synthetic_data()
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")
    
    # 2. Preprocess Data
    print("Preprocessing text data...")
    df_clean = preprocess_df(df)
    
    X = df_clean['cleaned_text']
    y = df_clean['category']
    
    # 3. Train-Test Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. TF-IDF Vectorization
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = fit_tfidf(X_train_raw)
    
    X_train = vectorizer.transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    print(f"Vocabulary Size: {len(vectorizer.vocabulary_)}")
    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 5. Train & Compare Models
    # Model A: Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_results = evaluate_model(nb_model, X_test, y_test, "Naive Bayes (Multinomial)")
    
    # Model B: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # 6. Select and Save Best Model
    os.makedirs('models', exist_ok=True)
    
    # Determine the best model based on F1-Score
    if lr_results['f1'] >= nb_results['f1']:
        best_model_name = "Logistic Regression"
        best_model = lr_model
    else:
        best_model_name = "Naive Bayes"
        best_model = nb_model
        
    print(f"\n[INFO] Saving the best performing model ({best_model_name})...")
    
    # Save vectorizer
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    # Save model
    with open('models/email_classifier.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    print("Model and Vectorizer saved successfully in the 'models/' folder!")

if __name__ == '__main__':
    # Add src to python path if run directly
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_pipeline()
