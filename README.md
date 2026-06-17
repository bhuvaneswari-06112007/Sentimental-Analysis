Act as an expert Data Scientist and NLP Engineer. I want to build an end-to-end AI-Based Email Classification System based on the following specific requirements. Please provide a production-ready blueprint, including structured Python code (using pandas, scikit-learn, and Streamlit), for the entire pipeline. 

Here are the requirements for each phase:

1. DATASET ANALYSIS & PREPROCESSING: 
Assume we are using a standard multi-class Kaggle email dataset (e.g., categories like HR, Finance, IT Support). Write a clean preprocessing function that:
- Handles missing values and duplicates.
- Lowercases the text, removes special characters, tokens, and stop-words.
- Uses TF-IDF Vectorization to prepare the text (Subject + Body combined) for machine learning.

2. MODEL TRAINING & EVALUATION:
- Implement a training pipeline that compares two models: Naive Bayes and Logistic Regression.
- Split the data into training and testing sets.
- For evaluation, provide code to calculate and print Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

3. OPTIONAL ENHANCEMENT (Priority Detection):
- Show how the model or a secondary rule-based/ML system can flag emails as 'Low', 'Medium', or 'High' priority based on keywords or text urgency.

4. WEB DASHBOARD:
- Write a clean, self-contained Streamlit application code block. 
- The dashboard must include: text inputs for "Email Subject" and "Email Content", a "Predict" button, a display for the predicted category, a confidence score, and a mock history log of previous predictions.


