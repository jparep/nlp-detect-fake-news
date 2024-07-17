# ml/scripts/hyperparameter_tuning.py

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from utils import load_pickle
import config
from typing import Optional
import numpy as np

def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray) -> Optional[DecisionTreeClassifier]:
    """Perform hyperparameter tuning using RandomizedSearchCV on a pre-trained model if available."""
    try:
        # Load pre-trained model
        pre_trained_model = load_pickle(config.MODEL_PATHS['decision_tree'])
        print("Pre-trained model loaded successfully.")
        
        param_dist = {
            'max_depth': [None, 10, 30, 60],
            'min_samples_split': [2, 5, 9, 11]
        }
        
        model = RandomizedSearchCV(pre_trained_model, 
                                   param_distributions=param_dist, 
                                   n_iter=12, 
                                   cv=10, 
                                   n_jobs=-1, 
                                   verbose=1, 
                                   random_state=config.RANDOM_SEED)
        model.fit(X_train, y_train)
        
        # Save the best estimator found by the search
        best_estimator = model.best_estimator_
        return best_estimator

    except FileNotFoundError as e:
        print(f"File not found: {e}. Please ensure that the model and vectorizer are trained and saved before running hyperparameter tuning.")
        return None
    except Exception as e:
        print(f"An error occurred during hyperparameter tuning: {e}")
        return None
