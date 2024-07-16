# ml/scripts/train_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import config
from utils import save_pickle

def train_model(X_train, y_train, model):
    """Train the model."""
    model.fit(X_train, y_train)
    return model

def train_and_save_models(X_train, y_train):
    """Train and save multiple models."""
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=config.RANDOM_SEED),
        'Decision Tree': DecisionTreeClassifier(random_state=config.RANDOM_SEED)
    }

    # Train and save models
    for name, model in models.items():
        print(f"Training {name}...")
        trained_model = train_model(X_train, y_train, model)
        model_path = config.MODEL_PATHS[name.replace(' ', '_').lower()]
        save_pickle(trained_model, model_path)
