import os

# Set the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to data files
FAKE_CSV_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'fake.csv')
REAL_CSV_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'true.csv')

# Path to model files
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'model', 'vectorizer.pkl')

# Random seed for reproducibility
RANDOM_SEED = 123
