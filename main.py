from ml.scripts.utils import load_and_preprocess_data, train_valid_test_split, vectorize_data
from ml.scripts.train_model import train_and_save_models
from ml.scripts.hyperparameter_tuning import hyperparameter_tuning_and_save
from ml.scripts.evaluate_model import evaluate_and_plot
import config

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    X = df['text']
    y = df['label']

    # Split data into train, validation, and test sets
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)

    # Vectorize data
    xv_train, xv_valid, xv_test, vectorizer = vectorize_data(X_train, X_valid, X_test, config.VECTORIZER_PATH)

    # Train and save models
    train_and_save_models(xv_train, y_train)

    # Perform hyperparameter tuning and save the best model
    hyperparameter_tuning_and_save(X_train, y_train)

    # Evaluate models and plot results
    evaluate_and_plot(xv_train, y_train, xv_valid, y_valid, xv_test, y_test)

if __name__ == "__main__":
    main()
