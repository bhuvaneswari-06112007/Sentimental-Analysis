import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Fallback set of English stop words in case NLTK download is slow/unavailable
DEFAULT_STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'cant', 'cannot', 'could',
    'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here',
    'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in',
    'into', 'is', 'isnt', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', 'no', 'nor',
    'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 'such', 'than', 'that', 'thats',
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they', 'theyd', 'theyll',
    'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasnt', 'we',
    'wed', 'well', 'were', 'weve', 'werent', 'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while',
    'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 'youll', 'youre', 'youve',
    'your', 'yours', 'yourself', 'yourselves'
}

def download_nltk_resources():
    """Download NLTK stopwords package dynamically, falling back silently if there's no connection."""
    try:
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except Exception:
        print("[WARNING] NLTK download failed or offline. Using default built-in stopwords list.")
        return DEFAULT_STOP_WORDS

# Load stopwords once
STOP_WORDS = download_nltk_resources()

def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    - Lowercasing
    - Removing special characters and numbers
    - Tokenizing
    - Removing stop-words
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove email patterns and URLs if present
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters, punctuation, and numbers (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize by splitting on whitespace
    tokens = text.split()
    
    # Filter out short tokens and stopwords
    cleaned_tokens = [word for word in tokens if len(word) > 1 and word not in STOP_WORDS]
    
    return " ".join(cleaned_tokens)

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the email dataframe:
    - Handles missing values and duplicates.
    - Combines Subject and Body.
    - Cleans combined text.
    """
    df = df.copy()
    
    # Handle missing values
    df['subject'] = df['subject'].fillna("")
    df['body'] = df['body'].fillna("")
    
    # Remove duplicate records
    df = df.drop_duplicates(subset=['subject', 'body']).reset_index(drop=True)
    
    # Combine Subject and Body
    df['combined_text'] = df['subject'] + " " + df['body']
    
    # Apply text cleaning
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    return df

def fit_tfidf(texts: list, max_features: int = 5000) -> TfidfVectorizer:
    """
    Fits a TF-IDF vectorizer on the given clean texts.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    vectorizer.fit(texts)
    return vectorizer
