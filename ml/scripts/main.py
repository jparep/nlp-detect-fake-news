from prepare_data import load_and_preprocess_data, train_valid_test_split, vectorize_data
from training_model import train_model
from evaluate_model import evaluate_model, plot_confusion_matrix
from utils import save_pickle, load_pickle
import config

def main():
    """Main function to execute the model training, tuning, and evaluation."""
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(config.REAL_CSV_PATH, config.FAKE_CSV_PATH)
        if df is None:
            raise ValueError("Data loading and preprocessing failed.")
        
        X = df['text']
        y = df['label']

        # Split data into train, validation, and test sets
        X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)
        if any([X_train is None, X_valid is None, X_test is None, y_train is None, y_valid is None, y_test is None]):
            raise ValueError("Data splitting failed.")

        # Vectorize data
        xv_train, xv_valid, xv_test, vectorizer = vectorize_data(X_train, X_valid, X_test)
        if any([xv_train is None, xv_valid is None, xv_test is None]):
            raise ValueError("Data vectorization failed.")
        save_pickle(vectorizer, config.VECTORIZER_PATH)

        # Train and save models
        trained_model = train_model(xv_train, y_train)
        if trained_model is None:
            raise ValueError("Model training failed.")
        save_pickle(trained_model, config.MODEL_PATH)

        # Load the trained model
        model = load_pickle(config.MODEL_PATH)
        if model is None:
            raise ValueError("Model loading failed.")

        # Predict and evaluate
        y_train_pred = model.predict(xv_train)
        y_valid_pred = model.predict(xv_valid)
        y_test_pred = model.predict(xv_test)
        
        print('Training Model:')
        evaluate_model(y_train, y_train_pred)
        print("\nEvaluating Model:")
        evaluate_model(y_valid, y_valid_pred)
        print("\nTesting Model: ")
        evaluate_model(y_test, y_test_pred)
        
        plot_confusion_matrix(y_test, y_test_pred)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
