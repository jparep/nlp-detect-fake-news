# Import libraries
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from data_preparation import load_and_preprocess_data, train_valid_test_split
from model_training import vectorize_data

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate the modela dn print metics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"ROC AUC Score: {roc_auc}")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    disp.plot()
    plt.show()

def load_model(model_path):
    """Load a saved model"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    df = load_d