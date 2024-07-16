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
MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'models')

# Path to individual model files
MODEL_PATHS = {
    'random_forest': os.path.join(MODEL_DIR, 'random_forest.pkl'),
    'decision_tree': os.path.join(MODEL_DIR, 'decision_tree.pkl'),
    'optimized_model': os.path.join(MODEL_DIR, 'optimized_model.pkl')
}

# Path to vectorizer
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

# Make sure directory exisist
def ensure_dir(path):
    """Ensure the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
