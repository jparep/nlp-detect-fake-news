# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import config
from data_preparation import load_and_preprocess_data, train_valid_test_split

def vectorize_data(X_train, X_valid, X_test, vectorizer_path):
    """Vectorize the text data using TF-IDF."""
    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(X_train)
    xv_valid = vectorizer.transform(X_valid)  # Corrected to use transform
    xv_test = vectorizer.transform(X_test)    # Corrected to use transform
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    return xv_train, xv_valid, xv_test, vectorizer

def train_model(X_train, y_train, model):
    """Train the model."""
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """Save the trained model."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    X = df['text']
    y = df['label']
    
    # Split data into train, validation, and test sets
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)
    
    # Vectorize data
    xv_train, xv_valid, xv_test, vectorizer = vectorize_data(X_train, X_valid, X_test, config.VECTORIZER_PATH)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=config.RANDOM_SEED),
        'Decision Tree': DecisionTreeClassifier(random_state=config.RANDOM_SEED)
    }
    
    # Train and save models
    for name, model in models.items():
        print(f"Training {name}...")
        model_trained = train_model(xv_train, y_train, model)
        save_model(model_trained, f"/home/jparep/proj/nlp-detect-fake-news/ml/model{name.replace(' ', '_').lower()}.pkl")

if __name__ == "__main__":
    main()
