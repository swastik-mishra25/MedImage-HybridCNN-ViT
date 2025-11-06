import torch
from src.utils import evaluate_metrics
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, val_dl, class_names, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm, report = evaluate_metrics(all_labels, all_preds, class_names)
    print("\nClassification Report:\n", report)

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrix.png")
    plt.show()
