#!/bin/bash
# CalorieCam Lite - Quick Setup and Run Script

echo "🍽️ CalorieCam Lite - Setting up and running..."
echo "=============================================="

# Navigate to project directory
cd "/Volumes/Sarvesh SSD/code/CalorieCam-Lite"

# Activate virtual environment
source ".venv/bin/activate"

echo "✅ Virtual environment activated"
echo "✅ Python version: $(python --version)"

# Check if required files exist
if [ ! -f "artifacts/base_model/best.pt" ]; then
    echo "❌ Model file not found. Creating dummy model..."
    python -c "
import torch, sys, os
sys.path.append('.')
from src.model import EmbeddingClassifier
model = EmbeddingClassifier(num_classes=5)
torch.save(model.state_dict(), 'artifacts/base_model/best.pt')
print('✅ Created dummy model file')
"
else
    echo "✅ Model file found"
fi

if [ ! -f "artifacts/base_model/label_map.json" ]; then
    echo "❌ Label map not found"
else
    echo "✅ Label map found"
fi

echo ""
echo "🚀 Starting Streamlit app..."
echo "   URL: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

# Run Streamlit with better configuration
streamlit run app/streamlit_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats False --theme.base dark
