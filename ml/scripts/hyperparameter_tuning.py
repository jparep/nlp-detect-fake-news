# ml/scripts/hyperparameter_tuning.py

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from typing import Optional

RANDOM_SEED = 42  # Replace with your actual random seed value

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV on a fresh DecisionTreeClassifier model."""
    try:
        classifier = DecisionTreeClassifier(random_state=RANDOM_SEED)

        param_dist = {
            'max_depth': [None, 10, 30, 60],
            'min_samples_split': [2, 5, 9, 11]
        }

        model = RandomizedSearchCV(estimator=classifier,
                                   param_distributions=param_dist,
                                   n_iter=12,
                                   cv=10,
                                   n_jobs=-1,
                                   verbose=1,
                                   random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        # Save the best estimator found by the search
        best_estimator = model.best_estimator_
        return best_estimator

    except Exception as e:
        print(f"An error occurred during hyperparameter tuning: {e}")
        return None