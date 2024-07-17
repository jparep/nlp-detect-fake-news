# ml/scripts/train_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import config
from utils import save_pickle

def train_model(X_train, y_train, model):
    """Train the model."""
    model.fit(X_train, y_train)
    return model

def train_and_save_models(X_train, y_train, model_paths):
    """Train and save multiple models."""
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=config.RANDOM_SEED),
        'Decision Tree': DecisionTreeClassifier(random_state=config.RANDOM_SEED)
    }

    # Train and save models
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            trained_model = train_model(X_train, y_train, model)
            save_pickle(trained_model, model_paths[name.replace(' ', '_').lower()])
            print(f"{name} model successfully saved to {model_paths[name.replace(' ', '_').lower()]}")
        except Exception as e:
            print(f"Error training and saving {name}: {e}")
