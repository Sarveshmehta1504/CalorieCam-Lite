#!/usr/bin/env python3
"""
Example usage of CalorieCam Lite with Kaggle dataset
Demonstrates the enhanced food recognition capabilities
"""
import sys
sys.path.append('.')
import pandas as pd
from PIL import Image
import torch
from src.model import EmbeddingClassifier
from src.data import get_transforms
from src.utils import load_json, device_auto

def demo_food_lookup():
    """Demonstrate the comprehensive food database"""
    print("🍽️ CalorieCam Lite - Food Database Demo")
    print("=" * 50)
    
    # Load the enhanced food database
    df = pd.read_csv('data/calorie_map.csv')
    
    print(f"📊 Total foods in database: {len(df)}")
    
    # Show some interesting categories
    categories = {
        '🍕 Pizza': 'pizza',
        '🍔 Burgers': 'burger', 
        '🍗 Chicken': 'chicken',
        '🍎 Apples': 'apple',
        '🍚 Rice': 'rice',
        '🥗 Salads': 'salad',
        '🍜 Soups': 'soup',
        '🍦 Ice Cream': 'ice cream'
    }
    
    print("\n🏷️ Food Categories Available:")
    for emoji_name, search_term in categories.items():
        count = len(df[df['label'].str.contains(search_term, case=False)])
        if count > 0:
            print(f"   {emoji_name}: {count} varieties")
    
    # Show some specific examples
    print(f"\n🔍 Sample Food Items:")
    sample_foods = ['pizza', 'cheeseburger', 'apple', 'chicken breast', 'chocolate cake']
    
    for food in sample_foods:
        matches = df[df['label'].str.contains(food, case=False)]
        if len(matches) > 0:
            item = matches.iloc[0]
            print(f"   • {item['label'].title()}: {item['avg_calories_kcal']:.0f} kcal ({item['portion_notes']})")

def demo_prediction():
    """Demonstrate food prediction with the test image"""
    print(f"\n🔍 Food Recognition Demo:")
    print("-" * 30)
    
    # Load model
    device = device_auto()
    label_map = load_json('artifacts/base_model/label_map.json')
    model = EmbeddingClassifier(num_classes=len(label_map)).to(device)
    model.load_state_dict(torch.load('artifacts/base_model/best.pt', map_location=device))
    model.eval()
    
    # Load test image
    img = Image.open('test_pizza.jpg')
    transform = get_transforms(train=False, img_size=224)
    x = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        logits, emb = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs.max().item()
    
    # Get prediction details
    inv_label = {int(k):v for k,v in label_map.items()}
    predicted_food = inv_label[pred_idx]
    
    print(f"📸 Image: test_pizza.jpg")
    print(f"🎯 Predicted: {predicted_food.title()}")
    print(f"📊 Confidence: {confidence:.1%}")
    
    # Lookup calories
    df = pd.read_csv('data/calorie_map.csv')
    calorie_info = df[df['label'] == predicted_food]
    
    if len(calorie_info) > 0:
        calories = calorie_info.iloc[0]['avg_calories_kcal']
        portion = calorie_info.iloc[0]['portion_notes']
        print(f"🔥 Calories: {calories:.0f} kcal")
        print(f"📏 Portion: {portion}")
    else:
        print("⚠️  Calorie info not found in database")

def main():
    """Run the demo"""
    try:
        demo_food_lookup()
        demo_prediction()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"\n🚀 To use the web interface:")
        print(f"   1. Open: http://localhost:8501")
        print(f"   2. Upload any food image")
        print(f"   3. Get instant predictions and calorie estimates")
        print(f"   4. View Grad-CAM explanations")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
