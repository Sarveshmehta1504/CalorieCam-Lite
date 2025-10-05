#!/usr/bin/env python3
"""
Combine real downloaded images with synthetic images and train the model
"""
import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.append('.')
from src.model import EmbeddingClassifier
from src.utils import device_auto
import json

def combine_datasets():
    """Combine real and synthetic food images"""
    print("🔄 Combining real and synthetic datasets...")
    
    # Create combined dataset directory
    combined_dir = Path("data/combined_food")
    combined_dir.mkdir(exist_ok=True)
    
    train_dir = combined_dir / "train"
    val_dir = combined_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    food_classes = ["pizza", "burger", "biryani", "dosa", "idly"]
    
    for food_class in food_classes:
        # Create class directories
        (train_dir / food_class).mkdir(exist_ok=True)
        (val_dir / food_class).mkdir(exist_ok=True)
        
        # Copy from real images (if exist)
        real_train = Path("data/food_images/train") / food_class
        real_val = Path("data/food_images/val") / food_class
        
        if real_train.exists():
            for img in real_train.glob("*.jpg"):
                shutil.copy2(img, train_dir / food_class / f"real_{img.name}")
        
        if real_val.exists():
            for img in real_val.glob("*.jpg"):
                shutil.copy2(img, val_dir / food_class / f"real_{img.name}")
        
        # Copy from synthetic images
        synth_train = Path("data/synthetic_food/train") / food_class
        synth_val = Path("data/synthetic_food/val") / food_class
        
        if synth_train.exists():
            for img in synth_train.glob("*.jpg"):
                shutil.copy2(img, train_dir / food_class / f"synth_{img.name}")
        
        if synth_val.exists():
            for img in synth_val.glob("*.jpg"):
                shutil.copy2(img, val_dir / food_class / f"synth_{img.name}")
        
        # Count images
        train_count = len(list((train_dir / food_class).glob("*.jpg")))
        val_count = len(list((val_dir / food_class).glob("*.jpg")))
        
        print(f"   {food_class}: {train_count} train, {val_count} val images")
    
    print("✅ Dataset combination completed")
    return combined_dir

def create_data_loaders(data_dir, batch_size=16):
    """Create data loaders for training"""
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=data_dir / "train",
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=data_dir / "val", 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes

def train_model(data_dir, epochs=10, lr=0.001):
    """Train the food recognition model"""
    print("🚀 Starting model training...")
    
    device = device_auto()
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, classes = create_data_loaders(data_dir, batch_size=16)
    
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = EmbeddingClassifier(num_classes=len(classes)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            if batch_idx % 5 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}")
        
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                logits, _ = model(data)
                loss = criterion(logits, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "artifacts/base_model/best.pt")
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
    
    # Save final model and metadata
    torch.save(model.state_dict(), "artifacts/base_model/last.pt")
    
    # Save label map
    label_map = {str(i): class_name for i, class_name in enumerate(classes)}
    with open("artifacts/base_model/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: artifacts/base_model/best.pt")
    
    return model, best_val_acc

def test_trained_model():
    """Test the trained model with our test image"""
    print("\n🧪 Testing trained model...")
    
    device = device_auto()
    
    # Load model
    with open("artifacts/base_model/label_map.json", "r") as f:
        label_map = json.load(f)
    
    model = EmbeddingClassifier(num_classes=len(label_map)).to(device)
    model.load_state_dict(torch.load("artifacts/base_model/best.pt", map_location=device))
    model.eval()
    
    # Load and preprocess test image
    from PIL import Image
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open("test_pizza.jpg").convert("RGB")
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        logits, _ = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs.max().item()
    
    predicted_class = label_map[str(pred_idx)]
    
    print(f"📸 Test image: test_pizza.jpg")
    print(f"🎯 Predicted: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2%}")
    
    # Show all class probabilities
    print(f"\n📋 All class probabilities:")
    for i, prob in enumerate(probs[0]):
        class_name = label_map[str(i)]
        print(f"   {class_name}: {prob:.2%}")

def main():
    """Main training pipeline"""
    print("🍽️ CalorieCam Lite - Model Training Pipeline")
    print("=" * 60)
    
    # Step 1: Combine datasets
    combined_dir = combine_datasets()
    
    # Step 2: Train model
    model, best_acc = train_model(combined_dir, epochs=15, lr=0.001)
    
    # Step 3: Test model
    test_trained_model()
    
    print(f"\n🎊 Training pipeline completed!")
    print(f"🏆 Best accuracy achieved: {best_acc:.2f}%")
    print(f"🌐 Your Streamlit app is now ready with a trained model!")
    print(f"🚀 Run: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()
