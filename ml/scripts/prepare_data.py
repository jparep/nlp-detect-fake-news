import os
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import save_pickle
import config

# Download necessary NLTK data (uncomment if running for the first time)
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Set random seed for reproducibility
np.random.seed(config.RANDOM_SEED)

def load_data(path):
    """Load data from a CSV file."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        data = pd.read_csv(path)
        if data.empty:
            raise pd.errors.EmptyDataError(f"No data in file: {path}")
        print(f"Data loaded from {path}:")
        print(data.head())  # Print the first few rows for debugging
        return data
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return None
    except pd.errors.EmptyDataError as ede_error:
        print(ede_error)
        return None
    except Exception as e:
        print(f"An error occurred while loading data from {path}: {e}")
        return None

def load_and_concatenate_data(real_csv, fake_csv):
    """Load and concatenate real and fake news data."""
    try:
        df_real = load_data(real_csv)
        df_fake = load_data(fake_csv)
        if df_real is None or df_fake is None:
            raise ValueError("One of the data files is missing or empty.")
        print(f"Real data shape: {df_real.shape}")
        print(f"Fake data shape: {df_fake.shape}")
        df_real['label'] = 'real'
        df_fake['label'] = 'fake'
        df_combine = pd.concat([df_real, df_fake], axis=0).sample(frac=1).reset_index(drop=True)
        print(f"Combined data shape: {df_combine.shape}")
        return df_combine
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_text(text):
    """Preprocess text data: remove non-alphabetical characters, lowercase, tokenize, lemmatize, remove stopwords."""
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text).lower()
    word_tokens = word_tokenize(text)
    lem = [lemmatizer.lemmatize(word) for word in word_tokens if word not in stop_words]
    return ' '.join(lem)

def load_and_preprocess_data(real_path, fake_path):
    """Load, concatenate, and preprocess data."""
    try:
        df = load_and_concatenate_data(real_path, fake_path)
        if df is None:
            return None
        df = df[['text', 'label']]
        print(f"Data before preprocessing: {df.head()}")
        df['text'] = df['text'].apply(preprocess_text)
        df['label'] = df['label'].map({'real': 0, 'fake': 1})
        print(f"Data after preprocessing: {df.head()}")
        return df
    except Exception as e:
        print(f"Error loading and preprocessing data: {e}")
        return None

def train_valid_test_split(X, y, train_size=0.7, valid_size=0.15, test_size=0.15):
    """Split data into train, validation, and test sets."""
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(valid_size + test_size), random_state=config.RANDOM_SEED)
        ratio = valid_size / (valid_size + test_size)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=(1.0 - ratio), random_state=config.RANDOM_SEED)
        print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}, Test shape: {X_test.shape}")
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None, None, None, None, None

def vectorize_data(X_train, X_valid, X_test, vectorizer_path):
    """Vectorize the text data using TF-IDF."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        vectorizer = TfidfVectorizer()
        xv_train = vectorizer.fit_transform(X_train)
        xv_valid = vectorizer.transform(X_valid)
        xv_test = vectorizer.transform(X_test)
        save_pickle(vectorizer, vectorizer_path)
        print(f"Vectorized data shapes - Train: {xv_train.shape}, Valid: {xv_valid.shape}, Test: {xv_test.shape}")
        return xv_train, xv_valid, xv_test, vectorizer
    except Exception as e:
        print(f"Error vectorizing data: {e}")
        return None, None, None, None
