# ml/scripts/config.py

import os

# Random seed for reproducibility
RANDOM_SEED = 123

# Set the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to data file directory
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Path to data files
FAKE_CSV_PATH = os.path.join(DATA_DIR, 'fake.csv')
REAL_CSV_PATH = os.path.join(DATA_DIR, 'true.csv')

# Path to model directory
MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'models', 'model.pkl')

# Path to vectorizer (not needed with BERT, but kept for completeness)
VECTORIZER_PATH = os.path.join(BASE_DIR, 'ml', 'models', 'vectorizer.pkl')
