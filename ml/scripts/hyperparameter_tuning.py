# ml/scripts/hyperparameter_tuning.py

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import config 
import numpy as np
from utils import load_pickle
from typing import Optional


def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray, vectorizer_path: str) -> Optional[Pipeline]:
    """Perform hyperparameter tuning using RandomizedSearchCV on a fresh DecisionTreeClassifier model."""
    
    try:
        # Load saved vectorizer
        vectorizer = load_pickle(vectorizer_path)
        print("Vectorizer loaded successfully.")
        
        # Create a pipeline with the loaded vectorizer and a new DecisionTreeClassifier
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('decision_tree', DecisionTreeClassifier(random_state=config.RANDOM_SEED))
        ])

        param_dist = {
            'decision_tree__max_depth': [None, 10, 30, 60],
            'decision_tree__min_samples_split': [2, 5, 9, 11]
        }

        model = RandomizedSearchCV(pipeline,
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
        print(f"File not found: {e}. Please ensure that the vectorizer is trained and saved before running hyperparameter tuning.")
        return None
    except Exception as e:
        print(f"An error occurred during hyperparameter tuning: {e}")
        return None
