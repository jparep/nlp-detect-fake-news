# Import necessary libraries
import numpy as np
import pandas as pd
import config
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Initalize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Set randomsed for reproducibility
np.random.seed(config.RANDOM_SEED)

# Load Data
def load_data(real_csv, fake_csv):
    """Load and concatenate real and fake news data."""
    df_real = pd.read_csv(real_csv)
    df_fake = pd.read_csv(fake_csv)
    df_combine = pd.concat([df_real, df_fake], axis=0).sample(frac=1).reset_index(drop=True)
    return df_combine

# Preprocess data
def preprocess_data(text):
    """Preprocess text data: remove non-alphaberical characters, lowercase, tokenize, lemmatize, remove stopwords"""
    


