import os
import pandas as pd
import joblib

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

def save_joblib(obj, path):
    """Save an object to a joblib file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(obj, path)
        print(f"Object successfully saved to {path}")
    except Exception as e:
        print(f"An error occurred while saving object to {path}: {e}")

def load_joblib(path):
    """Load an object from a joblib file."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
    except joblib.externals.loky.backend.exceptions.UnpicklingError:
        print(f"Unpickling error: {path}")
    except Exception as e:
        print(f"An error occurred while loading object from {path}: {e}")
