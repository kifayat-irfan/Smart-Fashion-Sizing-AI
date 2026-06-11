import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class FashionEvaluator:
    """
    Handles professional evaluation of the Multi-Output Model.
    """
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, target_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix: {target_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    @staticmethod
    def print_metrics(y_true, y_pred, class_names, target_name):
        print(f"\n--- {target_name} Report ---")
        print(classification_report(y_true, y_pred, target_names=class_names))
