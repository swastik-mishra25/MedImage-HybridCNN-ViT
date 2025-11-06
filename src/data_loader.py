import os, glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HistopathologyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def create_dataloaders(data_root, img_size=224, batch_size=8, val_split=0.2):
    classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    assert len(classes) == 2, "Binary classification requires exactly 2 folders under data_root"

    all_paths, all_labels = [], []
    for idx, cls in enumerate(classes):
        for f in glob.glob(os.path.join(data_root, cls, "*")):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                all_paths.append(f)
                all_labels.append(idx)

    train_x, val_x, train_y, val_y = train_test_split(
        all_paths, all_labels, test_size=val_split, stratify=all_labels, random_state=42
    )

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7, 0.5, 0.7], std=[0.2, 0.2, 0.2])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7, 0.5, 0.7], std=[0.2, 0.2, 0.2])
    ])

    train_ds = HistopathologyDataset(train_x, train_y, transform=train_tf)
    val_ds = HistopathologyDataset(val_x, val_y, transform=val_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_dl, val_dl, classes
