import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import config
from utils import save_pickle

def train_model(X_train, y_train):
    """Train and save multiple models."""
    # Define models
    model = DecisionTreeClassifier(random_state=config.RANDOM_SEED)
    # Train and save models
    model.fit(X_train, y_train)
    return model
