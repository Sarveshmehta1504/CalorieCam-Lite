# 🍽️ CalorieCam Lite - Setup & Usage Guide

## ✅ Virtual Environment Configuration Complete!

Your Python virtual environment is now fully configured and ready to use.

### 📋 Environment Details
- **Python Version**: 3.9.6
- **Virtual Environment Path**: `/Volumes/Sarvesh SSD/code/CalorieCam-Lite/.venv/`
- **All Required Packages**: ✅ Installed and working

### 🚀 How to Run the Application

#### Option 1: Quick Start (Recommended)
```bash
cd "/Volumes/Sarvesh SSD/code/CalorieCam-Lite"
./run_app.sh
```

#### Option 2: Manual Start
```bash
cd "/Volumes/Sarvesh SSD/code/CalorieCam-Lite"
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

#### Option 3: Direct Python Command
```bash
cd "/Volumes/Sarvesh SSD/code/CalorieCam-Lite"
"/Volumes/Sarvesh SSD/code/CalorieCam-Lite/.venv/bin/streamlit" run app/streamlit_app.py
```

### 🌐 Access the Application
- **URL**: http://localhost:8501
- **Status**: ✅ Currently running and accessible

### 📁 Project Structure Status
```
✅ Virtual Environment (.venv/)
✅ Source Code (src/)
✅ Streamlit App (app/streamlit_app.py)
✅ Model Artifacts (artifacts/base_model/)
✅ Sample Data (data/calorie_map.csv)
✅ Requirements (requirements.txt)
✅ Documentation (README.md)
```

### 🔧 Available Features

#### 1. **Image Upload & Prediction**
- Upload food images (JPG, PNG)
- Get calorie predictions
- View confidence scores

#### 2. **Two Inference Modes**
- **Prototype Mode**: Few-shot + base classes (nearest prototype)
- **Linear Head Mode**: Base classes only (standard classification)

#### 3. **Grad-CAM Explainability**
- Visual heatmaps showing what the model focuses on
- Helps understand model decisions

#### 4. **Calorie Estimation**
- Built-in calorie database for common foods
- Extensible via `data/calorie_map.csv`

### 🍕 Supported Food Classes (Default)
- Pizza
- Burger
- Biryani
- Dosa
- Idly

### 📊 Model Information
- **Architecture**: ResNet-18 with embedding layer
- **Embedding Dimension**: 512
- **Parameters**: ~11.2M
- **Pretrained**: ImageNet weights

### 🔄 Development Workflow

#### To Train Your Own Model:
```bash
# 1. Download dataset
python scripts/download_food101_subset.py --root data/raw --classes pizza burger biryani dosa idly

# 2. Train base model
python src/train.py --data_dir data/food_subset --epochs 10 --batch_size 32

# 3. Evaluate model
python src/eval.py --model_dir artifacts/base_model --data_dir data/food_subset
```

#### To Add Few-Shot Classes:
```bash
# Add new food class with 5-20 sample images
python src/adapt_fewshot.py --support_dir path/to/new_food_images --label new_food_name
```

### 🛠️ Troubleshooting

#### If the app doesn't start:
1. Check virtual environment: `source .venv/bin/activate`
2. Verify packages: `pip list | grep streamlit`
3. Check port availability: `lsof -i :8501`

#### If predictions fail:
1. Ensure model files exist in `artifacts/base_model/`
2. Check image format (JPG, PNG only)
3. Verify image size (recommended: 224x224 or larger)

### 📝 Notes
- The current setup includes dummy model weights for testing
- For production use, train the model with your own dataset
- Extend `data/calorie_map.csv` to add more food items and calorie values
- Model predictions are for educational/demo purposes

### 🎯 Next Steps
1. ✅ **Environment Setup** - Complete!
2. ✅ **App Running** - Ready to use!
3. 🔄 **Optional**: Train with real data
4. 🔄 **Optional**: Add more food classes
5. 🔄 **Optional**: Customize calorie database

**Happy Food Recognition! 🍽️**
