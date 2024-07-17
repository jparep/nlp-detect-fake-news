import pandas as pd
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(path)
        logging.info(f"Data successfully loaded from {path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
    except pd.errors.EmptyDataError:
        logging.error(f"No data: {path}")
    except Exception as e:
        logging.error(f"An error occurred while loading data from {path}: {e}")
    return None

def save_pickle(obj, path):
    """Save an object to a pickle file."""
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        logging.info(f"Object successfully saved to {path}")
    except Exception as e:
        logging.error(f"An error occurred while saving object to {path}: {e}")

def load_pickle(path):
    """Load an object from a pickle file."""
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logging.info(f"Object successfully loaded from {path}")
        return obj
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
    except pickle.UnpicklingError:
        logging.error(f"Unpickling error: {path}")
    except Exception as e:
        logging.error(f"An error occurred while loading object from {path}: {e}")
    return None
