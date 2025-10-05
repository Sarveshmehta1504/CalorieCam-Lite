import argparse, os, tarfile, requests, shutil, random
from pathlib import Path

FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"

def download_food101(root: str):
    os.makedirs(root, exist_ok=True)
    tgz_path = os.path.join(root, "food-101.tar.gz")
    if not os.path.exists(tgz_path):
        print("Downloading Food-101 (~1.3GB)...")
        with requests.get(FOOD101_URL, stream=True) as r:
            r.raise_for_status()
            with open(tgz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1<<20):
                    if chunk:
                        f.write(chunk)
    extract_dir = os.path.join(root, "food-101")
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=root)
    return extract_dir

def build_subset(food_root: str, classes: list, max_per_class: int, out_root: str):
    images_dir = os.path.join(food_root, "images")
    meta_dir = os.path.join(food_root, "meta")
    assert os.path.isdir(images_dir), "Images dir not found"
    os.makedirs(out_root, exist_ok=True)
    # Train/val split: 80/20
    for split in ["train", "val"]:
        for c in classes:
            os.makedirs(os.path.join(out_root, split, c), exist_ok=True)

    for c in classes:
        class_dir = os.path.join(images_dir, c)
        if not os.path.isdir(class_dir):
            raise SystemExit(f"Class '{c}' not found in Food-101. Check class list.")
        imgs = [p for p in os.listdir(class_dir) if p.endswith(".jpg")]
        random.shuffle(imgs)
        imgs = imgs[:max_per_class]
        n_train = int(0.8 * len(imgs))
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:]
        for fn in train_imgs:
            shutil.copy2(os.path.join(class_dir, fn), os.path.join(out_root,"train",c,fn))
        for fn in val_imgs:
            shutil.copy2(os.path.join(class_dir, fn), os.path.join(out_root,"val",c,fn))
        print(f"Class {c}: train {len(train_imgs)} | val {len(val_imgs)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw", type=str)
    ap.add_argument("--classes", nargs="+", required=True, help="Food-101 class names, e.g., dosa idly biryani pizza burger")
    ap.add_argument("--max_per_class", type=int, default=500)
    args = ap.parse_args()

    food_root = download_food101(args.root)
    out_root = os.path.join("data","food_subset")
    build_subset(food_root, args.classes, args.max_per_class, out_root)
    print("Done. Subset at:", out_root)

if __name__ == "__main__":
    main()
