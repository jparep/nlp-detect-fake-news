# Import necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import config

# Download necessary NLTK data (uncomment if running for the first time)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Set random seed for reproducibility
np.random.seed(config.RANDOM_SEED)

# Load Data
def load_data(real_csv, fake_csv):
    """Load and concatenate real and fake news data."""
    df_real = pd.read_csv(real_csv)
    df_fake = pd.read_csv(fake_csv)
    df_real['label'] = 'real'
    df_fake['label'] = 'fake'
    df_combine = pd.concat([df_real, df_fake], axis=0).sample(frac=1).reset_index(drop=True)
    return df_combine

# Preprocess data
def preprocess_text(text):
    """Preprocess text data: remove non-alphabetical characters, lowercase, tokenize, lemmatize, remove stopwords."""
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text).lower()
    word_tokens = word_tokenize(text)
    lem = [lemmatizer.lemmatize(word) for word in word_tokens if word not in stop_words]
    return ' '.join(lem)

def load_and_preprocess_data():
    """Load, concatenate, and preprocess data."""
    df = load_data(config.REAL_CSV_PATH, config.FAKE_CSV_PATH)
    df = df[['text', 'label']]
    df['text'] = df['text'].apply(preprocess_text)
    df['label'] = df['label'].map({'real': 0, 'fake': 1})
    return df

def train_valid_test_split(X, y, train_size=0.7, valid_size=0.15, test_size=0.15):
    """Split data into train, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(valid_size + test_size), random_state=config.RANDOM_SEED)
    ratio = valid_size / (valid_size + test_size)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=(1.0 - ratio), random_state=config.RANDOM_SEED)
    return X_train, X_valid, X_test, y_train, y_valid, y_test
