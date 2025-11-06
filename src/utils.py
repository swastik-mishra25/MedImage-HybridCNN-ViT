import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def accuracy(preds, labels):
    return (preds == labels).sum().item() / len(labels)

def evaluate_metrics(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    return cm, report
