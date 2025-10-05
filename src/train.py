import argparse, os, torch, torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from src.data import build_loaders
from src.model import EmbeddingClassifier
from src.utils import set_seed, device_auto, save_json

def train_one_epoch(model, loader, criterion, optim, device):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        n += y.size(0)
    return loss_sum/n, correct/n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        n += y.size(0)
    return loss_sum/n, correct/n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/food_subset")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=str, default="artifacts/base_model")
    args = ap.parse_args()

    set_seed(42)
    device = device_auto()
    train_loader, val_loader, classes = build_loaders(args.data_dir, args.batch_size)

    model = EmbeddingClassifier(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optim, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

    # Save label map
    label_map = {i:c for i,c in enumerate(classes)}
    save_json(label_map, os.path.join(args.out_dir, "label_map.json"))
    print("Done. Artifacts at:", args.out_dir)

if __name__ == "__main__":
    main()
