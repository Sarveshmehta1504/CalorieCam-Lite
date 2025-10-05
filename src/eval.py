import argparse, os, torch, torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd, numpy as np
from tqdm import tqdm
from src.data import build_loaders
from src.model import EmbeddingClassifier
from src.utils import load_json, device_auto

@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits.argmax(1)
        y_true += y.cpu().tolist()
        y_pred += pred.cpu().tolist()
    return np.array(y_true), np.array(y_pred)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/food_subset")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts/base_model")
    ap.add_argument("--ckpt", type=str, default="best.pt")
    args = ap.parse_args()

    device = device_auto()
    _, val_loader, classes = build_loaders(args.data_dir, batch_size=32)

    label_map = load_json(os.path.join(args.artifacts_dir,"label_map.json"))
    model = EmbeddingClassifier(num_classes=len(label_map)).to(device)
    model.load_state_dict(torch.load(os.path.join(args.artifacts_dir, args.ckpt), map_location=device))

    y_true, y_pred = evaluate_full(model, val_loader, device)
    inv = {int(k):v for k,v in label_map.items()}
    target_names = [inv[i] for i in sorted(inv.keys())]
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
