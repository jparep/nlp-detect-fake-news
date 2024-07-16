# ml/scripts/hyperparameter_tuning.py

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import pickle
from utils import load_pickle
import config

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV on a pre-trained model if available."""
    try:
        # Load pre-trained model and vectorizer
        pre_trained_model = load_pickle(config.MODEL_PATHS['decision_tree'])
        vectorizer = load_pickle(config.VECTORIZER_PATH)
        print("Pre-trained model and vectorizer loaded successfully.")
        
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', pre_trained_model)
        ])
        
        param_dist = {
            'classifier__max_depth': [None, 10, 30, 60],
            'classifier__min_samples_split': [2, 5, 9, 11]
        }
        
        model = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=12, cv=10, n_jobs=-1, verbose=1, random_state=config.RANDOM_SEED)
        model.fit(X_train, y_train)
        
        # Save the best estimator found by the search
        best_estimator = model.best_estimator_
        return best_estimator

    except FileNotFoundError as e:
        print(f"File not found: {e}. Please ensure that the model and vectorizer are trained and saved before running hyperparameter tuning.")
        return None
