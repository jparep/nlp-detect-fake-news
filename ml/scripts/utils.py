# ml/scripts/utils.py

import pandas as pd
import pickle

def load_data(path):
    """Load data from a CSV file."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
    except pd.errors.EmptyDataError:
        print(f"No data: {path}")
    except Exception as e:
        print(f"An error occurred while loading data from {path}: {e}")

def save_pickle(obj, path):
    """Save an object to a pickle file."""
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Object successfully saved to {path}")
    except Exception as e:
        print(f"An error occurred while saving object to {path}: {e}")

def load_pickle(path):
    """Load an object from a pickle file."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
    except pickle.UnpicklingError:
        print(f"Unpickling error: {path}")
    except Exception as e:
        print(f"An error occurred while loading object from {path}: {e}")
