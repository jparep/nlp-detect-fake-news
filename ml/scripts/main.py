# main.py

import os
from prepare_data import load_and_preprocess_data, train_valid_test_split, vectorize_data
from train_model import train_and_save_models
from hyperparameter_tuning import hyperparameter_tuning
from evaluate_model import evaluate_model, plot_confusion_matrix
from utils import save_pickle, load_pickle
import config as config

def main():
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        X = df['text']
        y = df['label']

        # Split data into train, validation, and test sets
        X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)

        # Vectorize data
        xv_train, xv_valid, xv_test = vectorize_data(X_train, X_valid, X_test, config.VECTORIZER_PATH)

        # Ensure the models directory exists
        config.ensure_dir(config.MODEL_DIR)

        # Train and save models
        train_and_save_models(xv_train, y_train)

        # Perform hyperparameter tuning and save the best model
        optimized_model = hyperparameter_tuning(xv_train, y_train)
        save_pickle(optimized_model, config.MODEL_PATHS['optimized_model'])

        # Evaluate models and plot results
        for name, model_path in config.MODEL_PATHS.items():
            print(f"Evaluating {name.replace('_', ' ').title()}...")
            model = load_pickle(model_path)
            
            if name == 'optimized_model':
                y_train_pred = model.predict(X_train)
                y_valid_pred = model.predict(X_valid)
                y_test_pred = model.predict(X_test)
            else:
                y_train_pred = model.predict(xv_train)
                y_valid_pred = model.predict(xv_valid)
                y_test_pred = model.predict(xv_test)
            
            evaluate_model(y_train, y_train_pred, f"{name.replace('_', ' ').title()} (train)")
            evaluate_model(y_valid, y_valid_pred, f"{name.replace('_', ' ').title()} (valid)")
            evaluate_model(y_test, y_test_pred, f"{name.replace('_', ' ').title()} (test)")
            
            plot_confusion_matrix(y_test, y_test_pred)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
