import os

# Set the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to data files
FAKE_CSV_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'fake.csv')