import os
import pickle
import streamlit as st
import pandas as pd
from datetime import datetime

# Import modules from src
from src.preprocessing import clean_text
from src.priority import detect_priority

# Set Page Config
st.set_page_config(
    page_title="AI Email Classification & Priority Portal",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
    <style>
        /* Main page adjustments */
        .main {
            background-color: #f7f9fc;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        
        /* Banner Header */
        .header-container {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 30px;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Card-like containers for content */
        .card {
            background-color: white;
            padding: 24px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border-left: 5px solid #2a5298;
        }
        
        /* Badge styling */
        .badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            display: inline-block;
            margin-right: 5px;
            text-transform: uppercase;
        }
        
        .badge-hr { background-color: #ffe8e8; color: #d93838; border: 1px solid #ffccd2; }
        .badge-finance { background-color: #e2f9e6; color: #1e7e34; border: 1px solid #c3e6cb; }
        .badge-it { background-color: #e2f0fe; color: #004085; border: 1px solid #b8daff; }
        .badge-marketing { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
        
        .badge-high { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; font-weight: bold; }
        .badge-medium { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
        .badge-low { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        
        /* Metric styling */
        .metric-value {
            font-size: 2.2em;
            font-weight: 800;
            color: #1e3c72;
            margin-top: 10px;
        }
        
        .metric-label {
            font-size: 0.9em;
            text-transform: uppercase;
            color: #6c757d;
            letter-spacing: 1px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Helper function to get badge class based on category
def get_category_badge(category):
    cat_lower = category.lower()
    if 'hr' in cat_lower:
        return 'badge-hr'
    elif 'finance' in cat_lower:
        return 'badge-finance'
    elif 'it' in cat_lower:
        return 'badge-it'
    else:
        return 'badge-marketing'

# Helper function to get badge class based on priority
def get_priority_badge(priority):
    p_lower = priority.lower()
    if p_lower == 'high':
        return 'badge-high'
    elif p_lower == 'medium':
        return 'badge-medium'
    else:
        return 'badge-low'

# Initialize session state for prediction history
if 'history' not in st.session_state:
    st.session_state.history = []

# Page Title Layout
st.markdown("""
    <div class="header-container">
        <h1 style="margin: 0; font-size: 2.4em;">✉️ AI-Powered Email Classification Hub</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.85;">Intelligent Multi-Class Categorization & Priority Routing Engine</p>
    </div>
""", unsafe_allow_html=True)

# Check if models are available, if not, offer a direct training button
models_exist = os.path.exists('models/email_classifier.pkl') and os.path.exists('models/tfidf_vectorizer.pkl')

# Sidebar Controls
st.sidebar.title("🛠️ System Control Panel")

if not models_exist:
    st.sidebar.warning("⚠️ ML Model is not trained yet!")
    if st.sidebar.button("⚙️ Train ML Classifier Now"):
        with st.spinner("Initializing pipeline, generating dataset, and training model..."):
            from src.train import train_pipeline
            train_pipeline()
            st.sidebar.success("✅ Models trained and loaded successfully!")
            st.rerun()
else:
    st.sidebar.success("✅ ML Model & Vectorizer loaded.")
    if st.sidebar.button("🔄 Retrain ML Models"):
        with st.spinner("Retraining classifier..."):
            from src.train import train_pipeline
            train_pipeline()
            st.sidebar.success("✅ Models retrained successfully!")
            st.rerun()

# Prepopulate examples button for testing
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Load Test Email Examples")
examples = {
    "Select an example...": ("", ""),
    "IT Outage (High Priority)": (
        "CRITICAL: Production server down!",
        "Hi, Server-04 is completely offline. None of our users can access the database. Please resolve this ASAP! Urgent blocking issue."
    ),
    "HR Onboarding (Medium Priority)": (
        "Onboarding Document Request",
        "Hi Alex, welcome to the company! Please make sure to submit your signed contract and bank details for payroll by tomorrow so we can set up your account."
    ),
    "Finance Overdue Invoice (High Priority)": (
        "Invoice INV-2026-098 is overdue",
        "This is an urgent notice that payment for Invoice INV-2026-098 of $5,600 has not been received. Please process the transaction immediately to prevent service disruption."
    ),
    "Marketing Newsletter (Low Priority)": (
        "Summer Campaign Launch Newsletter",
        "Hi all, take a look at our brand new template and layout design guidelines for the newsletter scheduled for next month. Suggestions are welcome."
    )
}

selected_example = st.sidebar.selectbox("Choose a template:", list(examples.keys()))
default_subject, default_body = examples[selected_example]

# Main Layout splitting input and results
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="card"><h3>📥 Input Email Content</h3>', unsafe_allow_html=True)
    
    # Text Inputs
    subject_input = st.text_input("Email Subject", value=default_subject, placeholder="e.g., Urgent: VPN connection issue")
    body_input = st.text_area("Email Body", value=default_body, height=180, placeholder="Type or paste the email body here...")
    
    predict_btn = st.button("🔍 Run Classification Engine", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="card"><h3>📊 Analysis Output</h3>', unsafe_allow_html=True)
    
    if predict_btn or (default_subject != "" and default_body != ""):
        if not (subject_input.strip() or body_input.strip()):
            st.error("Please enter a Subject or Email Body to perform classification.")
        elif not models_exist:
            st.error("The ML Model is not trained. Please click 'Train ML Classifier Now' in the sidebar first.")
        else:
            with st.spinner("Analyzing email..."):
                # Load models
                with open('models/email_classifier.pkl', 'rb') as f:
                    model = pickle.load(f)
                with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                    vectorizer = pickle.load(f)
                
                # Combine and preprocess
                combined_text = f"{subject_input} {body_input}"
                cleaned_text = clean_text(combined_text)
                
                if not cleaned_text.strip():
                    st.warning("The email content is empty or contains only stopwords. Please write more context.")
                else:
                    # Vectorize
                    tfidf_features = vectorizer.transform([cleaned_text])
                    
                    # Predict Category & Probabilities
                    category = model.predict(tfidf_features)[0]
                    probabilities = model.predict_proba(tfidf_features)[0]
                    classes = model.classes_
                    
                    # Find confidence score for the predicted class
                    class_idx = list(classes).index(category)
                    confidence = probabilities[class_idx]
                    
                    # Priority Engine
                    priority_details = detect_priority(subject_input, body_input)
                    priority = priority_details['priority']
                    reasons = priority_details['reasons']
                    
                    # Store in history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_log = {
                        'Timestamp': timestamp,
                        'Subject': subject_input,
                        'Predicted Category': category,
                        'Confidence': f"{confidence:.2%}",
                        'Priority': priority
                    }
                    st.session_state.history.insert(0, new_log)
                    
                    # Display Results UI
                    col_metric1, col_metric2 = st.columns(2)
                    
                    with col_metric1:
                        badge_cls = get_category_badge(category)
                        st.markdown(f"""
                            <div class="metric-label">Predicted Category</div>
                            <div style="margin-top:10px;">
                                <span class="badge {badge_cls}" style="font-size:1.1em; padding: 8px 16px;">{category}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with col_metric2:
                        p_badge_cls = get_priority_badge(priority)
                        st.markdown(f"""
                            <div class="metric-label">Priority Status</div>
                            <div style="margin-top:10px;">
                                <span class="badge {p_badge_cls}" style="font-size:1.1em; padding: 8px 16px;">{priority}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Confidence Score Visual
                    st.markdown(f"<div class='metric-label'>Prediction Confidence Score</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{confidence:.2%}</div>", unsafe_allow_html=True)
                    st.progress(float(confidence))
                    
                    # Priority Reasoning
                    st.markdown("##### 🔍 Priority Assessment Notes")
                    for reason in reasons:
                        st.markdown(f"- {reason}")
    else:
        st.info("Enter subject/body content and press the 'Run Classification Engine' button to analyze.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# Bottom History Section
st.markdown("---")
st.subheader("📋 Session Prediction History Log")
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)
    if st.button("🧹 Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No predictions in this session yet. Make a prediction above to start logging.")
