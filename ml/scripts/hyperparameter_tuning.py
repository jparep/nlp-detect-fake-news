from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from utils import load_and_preprocess_data, train_valid_test_split, vectorize_data
import config
import pickle

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000)),
        ('classifier', DecisionTreeClassifier(random_state=config.RANDOM_SEED))
    ])
    
    param_dist = {
        'classifier__max_depth': [None, 10, 30, 60],
        'classifier__min_samples_split': [2, 5, 9, 11]
    }
    
    model = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=12, cv=10, n_jobs=-1, verbose=1, random_state=config.RANDOM_SEED)
    model.fit(X_train, y_train)
    return model.best_estimator_

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    X = df["text"]
    y = df["label"]
    
    # Split data into train, validation, and test sets
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)
    
    # Perform hyperparameter tuning
    optimized_model = hyperparameter_tuning(X_train, y_train)
    
    # Save the optimized model
    with open(config.MODEL_PATH, 'wb') as f:
        pickle.dump(optimized_model, f)
        
    print(f"Optimized model saved to {config.MODEL_PATH}")

if __name__ == '__main__':
    main()
