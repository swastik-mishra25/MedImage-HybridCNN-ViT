from src.model import HybridCNNViT
from src.data_loader import create_dataloaders
from src.train import train_model
from src.evaluate import evaluate_model
import torch

def main():
    # Path to your dataset (should contain: data/raw/benign/ and data/raw/malignant/)
    data_root = "data/raw"

    # Create dataloaders for training and validation
    train_dl, val_dl, classes = create_dataloaders(
        data_root, img_size=224, batch_size=8
    )

    # Initialize the Hybrid CNN-ViT model
    model = HybridCNNViT(num_classes=2)

    # Force the model to use CPU (prevents DLL issues on Windows)
    device = torch.device("cpu")
    print(f"✅ Using device: {device}")

    # Train the model
    train_model(model, train_dl, val_dl, device, epochs=5, lr=3e-4)

    # Evaluate the model
    evaluate_model(model, val_dl, classes, device)


if __name__ == "__main__":
    main()
