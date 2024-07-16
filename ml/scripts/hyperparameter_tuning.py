from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import config

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
