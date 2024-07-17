# ml/scripts/prepare_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
import re
import config
from utils import load_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize BERT tokenizer and model
try:
    logging.info("Loading BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    logging.info("BERT tokenizer and model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading BERT tokenizer or model: {e}")
    raise

def encode_texts(texts, tokenizer, model, max_length=512):
    """Encode texts using BERT and return embeddings."""
    try:
        inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
    except Exception as e:
        logging.error(f"Error encoding texts: {e}")
        return None

def vectorize_data(X_train, X_valid, X_test):
    """Vectorize text data using BERT."""
    try:
        xv_train = encode_texts(X_train, tokenizer, model)
        xv_valid = encode_texts(X_valid, tokenizer, model)
        xv_test = encode_texts(X_test, tokenizer, model)
        return xv_train, xv_valid, xv_test
    except Exception as e:
        logging.error(f"Error vectorizing data: {e}")
        return None, None, None

def preprocess_text(text):
    """Preprocess text data: remove non-alphabetical characters, lowercase."""
    try:
        text = re.sub(r'[^a-zA-Z0-9]+', ' ', text).lower()
        return text
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}")
        return ""

def load_and_preprocess_data():
    """Load, concatenate, and preprocess data."""
    try:
        df_real = load_data(config.REAL_CSV_PATH)
        df_fake = load_data(config.FAKE_CSV_PATH)
        df_real['label'] = 0
        df_fake['label'] = 1
        df = pd.concat([df_real, df_fake], axis=0).sample(frac=1).reset_index(drop=True)
        df['text'] = df['text'].apply(preprocess_text)
        return df
    except Exception as e:
        logging.error(f"Error loading and preprocessing data: {e}")
        return None

def train_valid_test_split(X, y, train_size=0.7, valid_size=0.15, test_size=0.15):
    """Split data into train, validation, and test sets."""
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(valid_size + test_size), random_state=config.RANDOM_SEED)
        ratio = valid_size / (valid_size + test_size)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=(1.0 - ratio), random_state=config.RANDOM_SEED)
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        return None, None, None, None, None, None
