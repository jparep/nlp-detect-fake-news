import os
from ml.scripts.prepare_data import load_and_preprocess_data, train_valid_test_split, vectorize_data
from ml.scripts.train_model import train_model
from ml.scripts.hyperparameter_tuning import hyperparameter_tuning
from ml.scripts.evaluate_model import evaluate_model, plot_confusion_matrix
from ml.scripts.utils import save_pickle, load_pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import ml.scripts.config as config

def main():
    try:
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
            trained_model = train_model(xv_train, y_train, model)
            model_path = os.path.join(config.BASE_DIR, 'ml', 'models', f"{name.replace(' ', '_').lower()}.pkl")
            save_pickle(trained_model, model_path)

        # Perform hyperparameter tuning and save the best model
        optimized_model = hyperparameter_tuning(X_train, y_train)
        optimized_model_path = os.path.join(config.BASE_DIR, 'ml', 'models', 'optimized_model.pkl')
        save_pickle(optimized_model, optimized_model_path)

        # Evaluate models and plot results
        for name in models.keys():
            model_path = os.path.join(config.BASE_DIR, 'ml', 'models', f"{name.replace(' ', '_').lower()}.pkl")
            model = load_pickle(model_path)
            y_train_pred = model.predict(xv_train)
            y_valid_pred = model.predict(xv_valid)
            y_test_pred = model.predict(xv_test)
            
            evaluate_model(y_train, y_train_pred, f"{name} (train)")
            evaluate_model(y_valid, y_valid_pred, f"{name} (valid)")
            evaluate_model(y_test, y_test_pred, f"{name} (test)")
            
            plot_confusion_matrix(y_test, y_test_pred)

        # Evaluate the optimized model
        optimized_model = load_pickle(optimized_model_path)
        y_train_pred = optimized_model.predict(xv_train)
        y_valid_pred = optimized_model.predict(xv_valid)
        y_test_pred = optimized_model.predict(xv_test)
        
        evaluate_model(y_train, y_train_pred, "Optimized Model (train)")
        evaluate_model(y_valid, y_valid_pred, "Optimized Model (valid)")
        evaluate_model(y_test, y_test_pred, "Optimized Model (test)")
        
        plot_confusion_matrix(y_test, y_test_pred)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
