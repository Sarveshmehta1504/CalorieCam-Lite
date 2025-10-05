import argparse, os, glob, torch
from PIL import Image
import numpy as np
from torchvision import transforms
from src.model import EmbeddingNet, EmbeddingClassifier
from src.utils import load_json, save_json, device_auto
from src.data import get_transforms

def image_paths(root):
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    return [p for p in glob.glob(os.path.join(root,"**","*"), recursive=True) if p.lower().endswith(exts)]

@torch.no_grad()
def compute_prototype(img_paths, embed_model, device, tform):
    embs = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        x = tform(img).unsqueeze(0).to(device)
        feat = embed_model(x).cpu().numpy()
        embs.append(feat[0])
    embs = np.stack(embs, axis=0)
    proto = embs.mean(axis=0).tolist()
    return proto, len(img_paths)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_dir", type=str, required=True, help="Folder with few-shot images for ONE class")
    ap.add_argument("--label", type=str, required=True, help="New class label, e.g., pani_puri")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts/base_model")
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    device = device_auto()

    # Load embedding model from the trained classifier backbone
    # We'll init a classifier to load weights, then take its embed part
    label_map = load_json(os.path.join(args.artifacts_dir,"label_map.json"))
    if label_map is None:
        raise SystemExit("label_map.json not found. Train base model first.")
    tmp = EmbeddingClassifier(num_classes=len(label_map)).to(device)
    ckpt = os.path.join(args.artifacts_dir, "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(args.artifacts_dir, "last.pt")
    tmp.load_state_dict(torch.load(ckpt, map_location=device))
    embed_model = tmp.embed
    embed_model.eval()

    tform = get_transforms(train=False, img_size=args.img_size)

    paths = image_paths(args.support_dir)
    if len(paths) < 3:
        raise SystemExit("Need at least 3 support images for a stable prototype.")
    proto, n = compute_prototype(paths, embed_model, device, tform)

    # Load existing prototypes or create new
    proto_path = os.path.join(args.artifacts_dir, "prototypes.json")
    protos = load_json(proto_path, default={"classes":{}, "meta":{}})
    protos["classes"][args.label] = {"vector": proto, "n_support": n}
    protos["meta"]["embed_dim"] = len(proto)
    save_json(protos, proto_path)
    print(f"Added/updated prototype for class '{args.label}' with {n} images at {proto_path}")

if __name__ == "__main__":
    main()
