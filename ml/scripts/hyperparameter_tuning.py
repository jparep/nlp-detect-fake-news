# ml/scripts/hyperparameter_tuning.py

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from utils import load_data, save_pickle, load_pickle  # Relative import
import config  # Updated import

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV on a pre-trained model if available."""
    # Load pre-trained model if it exists
    try:
        pre_trained_model = load_pickle(config.MODEL_PATHS['decision_tree'])
        print("Pre-trained model loaded successfully.")
    except FileNotFoundError:
        print("No pre-trained model found. Using a new DecisionTreeClassifier.")
        pre_trained_model = DecisionTreeClassifier(random_state=config.RANDOM_SEED)
    
    param_dist = {
        'max_depth': [None, 10, 30, 60],
        'min_samples_split': [2, 5, 9, 11]
    }
    
    model = RandomizedSearchCV(pre_trained_model, param_distributions=param_dist, n_iter=12, cv=10, n_jobs=-1, verbose=1, random_state=config.RANDOM_SEED)
    model.fit(X_train, y_train)
    
    return model.best_estimator_
