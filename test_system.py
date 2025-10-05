#!/usr/bin/env python3
"""
Test script for CalorieCam Lite functionality
"""
import sys
import os
sys.path.append('.')

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    try:
        import torch
        import torchvision
        import streamlit
        import numpy as np
        from PIL import Image
        import cv2
        
        from src.model import EmbeddingClassifier
        from src.data import get_transforms, load_calorie_map
        from src.utils import load_json, device_auto
        from src.explainability import GradCAM
        
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("\n🔍 Testing model loading...")
    try:
        import torch
        from src.utils import load_json, device_auto
        from src.model import EmbeddingClassifier
        
        device = device_auto()
        label_map = load_json("artifacts/base_model/label_map.json")
        
        if label_map is None:
            print("❌ No label map found")
            return False
            
        model = EmbeddingClassifier(num_classes=len(label_map))
        
        # Try to load model weights
        ckpt_path = "artifacts/base_model/best.pt"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            print(f"✅ Model loaded successfully! Classes: {list(label_map.values())}")
            return True
        else:
            print("❌ Model checkpoint not found")
            return False
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("\n🔍 Testing data loading...")
    try:
        from src.data import load_calorie_map
        
        cal_map = load_calorie_map("data/calorie_map.csv")
        print(f"✅ Calorie map loaded! Found {len(cal_map)} food items:")
        for food, calories in list(cal_map.items())[:3]:
            print(f"   - {food}: {calories} kcal")
        if len(cal_map) > 3:
            print(f"   ... and {len(cal_map)-3} more")
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_dummy_prediction():
    """Test dummy prediction"""
    print("\n🔍 Testing dummy prediction...")
    try:
        import torch
        import numpy as np
        from src.model import EmbeddingClassifier
        from src.data import get_transforms
        from PIL import Image
        
        # Create a dummy RGB image
        dummy_img = Image.new('RGB', (224, 224), color='red')
        
        # Transform to tensor
        transform = get_transforms(train=False, img_size=224)
        x = transform(dummy_img).unsqueeze(0)
        
        # Load model
        model = EmbeddingClassifier(num_classes=5)
        model.load_state_dict(torch.load("artifacts/base_model/best.pt", map_location='cpu'))
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            logits, embeddings = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs.max().item()
        
        print(f"✅ Prediction successful!")
        print(f"   Predicted class: {pred_class}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Embedding shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def main():
    """Run all tests"""
    print("🍽️ CalorieCam Lite - System Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_model_loading,
        test_data_loading,
        test_dummy_prediction
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To start the app, run:")
        print("   ./run_app.sh")
        print("   OR")
        print("   streamlit run app/streamlit_app.py")
    else:
        print("❌ Some tests failed. Please check the setup.")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()
