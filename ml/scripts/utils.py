# ml/scripts/utils.py

import pandas as pd
import pickle

def load_data(path):
    return pd.read_csv(path)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
