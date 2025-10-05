from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os, csv, torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(train=True, img_size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

def build_loaders(data_dir, batch_size=32, num_workers=2, img_size=224):
    train_ds = datasets.ImageFolder(os.path.join(data_dir,"train"), transform=get_transforms(True, img_size))
    val_ds   = datasets.ImageFolder(os.path.join(data_dir,"val"),   transform=get_transforms(False, img_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.classes

def load_calorie_map(csv_path):
    m = {}
    if not os.path.exists(csv_path):
        return m
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if i==0: continue
            label, kcal, notes = row
            try:
                m[label] = float(kcal)
            except:
                pass
    return m
