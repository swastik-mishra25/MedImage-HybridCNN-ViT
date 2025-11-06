import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils import accuracy

def train_model(model, train_dl, val_dl, device, epochs=5, lr=3e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0
        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            total_loss += loss.item()
            total_acc += accuracy(preds.cpu(), labels.cpu())

        print(f"Train Loss: {total_loss/len(train_dl):.4f} | Train Acc: {total_acc/len(train_dl):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = 0, 0
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(1)
                val_loss += loss.item()
                val_acc += accuracy(preds.cpu(), labels.cpu())
        print(f"Val Loss: {val_loss/len(val_dl):.4f} | Val Acc: {val_acc/len(val_dl):.4f}")

    torch.save(model.state_dict(), "outputs/checkpoints/best_model.pt")
    print("✅ Model saved at outputs/checkpoints/best_model.pt")
