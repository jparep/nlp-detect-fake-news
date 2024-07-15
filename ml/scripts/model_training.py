# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
from ml.scripts.data_preparation import load_and_preprocess_data, train_valid_test_split

def vectorize_data(X_train, X_valid, X_test):
    """Vectorize the text data using TF-IDF"""
    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(X_train)
    xv_valid = vectorizer.fit_transform(X_valid)
    xv_test = vectorizer.fit_transform(X_test)
    return xv_train, xv_valid, xv_test, vectorizer

def train_model(X_train, y_train, model):
    """Train the model"""
    model.fit(X_train, y_train)
    return model