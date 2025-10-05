# CalorieCam Lite — Few‑Shot Food Calorie Estimator (Python)

A graduation‑level, **fully runnable** ML project in Python that classifies food images and estimates calories.  
It supports **few‑shot adaptation** so you can add new dishes with as few as **5 images** (no full retraining).  
Includes a **Streamlit UI** and **Grad‑CAM** explainability.

---

## Features
- **Backbone:** ResNet‑18 (ImageNet) as an embedding network.
- **Base training:** Train a simple linear head on a small base dataset (e.g., Food‑101 subset).
- **Few‑shot adaptation:** Add new dishes by dropping 5–20 images in a folder. We build **class prototypes** (mean embeddings) and perform **nearest‑prototype** classification.
- **Calories:** Each class maps to a kcal estimate via `data/calorie_map.csv`. Few‑shot classes can add their own rows.
- **Explainability:** Grad‑CAM heatmaps overlayed on predictions in the Streamlit app.
- **Reproducible:** Deterministic seeds, clear file structure, minimal dependencies.

---

## Quick Start

### 0) Setup
```bash
# (Recommended) Create a virtual environment
python -m venv .venv && source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1) Get a base dataset
You can use a small subset of **Food‑101** (free). Run the helper script:
```bash
python scripts/download_food101_subset.py --root data/raw --classes pizza burger biryani dosa idly --max_per_class 500
```
This downloads Food‑101 and creates a subset under `data/food_subset/{train,val}` with the listed classes.

> Tip: Replace classes with any you want that exist in Food‑101 (e.g., pizza, burger, fries, samosa not in Food‑101; use dosa/idly/biryani/chapati etc. See Food‑101 class list).

### 2) Train a base classifier
```bash
python src/train.py --data_dir data/food_subset --epochs 5 --batch_size 32 --lr 1e-3 --out_dir artifacts/base_model
```

### 3) (Optional) Few‑shot adapt with your own dish
Create a folder with 5–20 images of the new dish:
```
fewshot/
  pani_puri/
    img1.jpg ... img10.jpg
```
Then run:
```bash
python src/adapt_fewshot.py --support_dir fewshot/pani_puri --label pani_puri --artifacts_dir artifacts/base_model
```
This updates `artifacts/base_model/prototypes.json` with a new class prototype.

### 4) Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```
Upload an image or select a sample, choose **Prototype (few‑shot)** or **Linear Head** mode, and view predictions + calories + Grad‑CAM.

---

## Project Structure
```
CalorieCam-Lite/
  app/
    streamlit_app.py
  data/
    calorie_map.csv
  scripts/
    download_food101_subset.py
  src/
    data.py
    model.py
    utils.py
    train.py
    eval.py
    adapt_fewshot.py
    explainability.py
  artifacts/            # created after training (weights, prototypes)
  requirements.txt
  README.md
```

---

## Notes
- The few‑shot classifier uses **nearest-prototype** over embeddings. It combines base classes (from linear head) and few‑shot classes (from prototypes) if both are present.
- Calories are estimates. Update `data/calorie_map.csv` to match your cuisine/portion assumptions.
- If GPU is available, training will use it automatically.

Enjoy, and feel free to extend!
