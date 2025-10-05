#!/usr/bin/env python3
"""
Download food images from the internet for training
Alternative approach when Food-101 dataset is not available
"""
import os
import requests
import time
import random
from pathlib import Path
from PIL import Image
import hashlib
from urllib.parse import urlparse

class InternetFoodImageDownloader:
    def __init__(self, base_dir="data/food_images"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-collected image URLs for different food categories
        self.food_urls = {
            "pizza": [
                "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=400",
                "https://images.unsplash.com/photo-1506354666786-959d6d497f1a?w=400",
                "https://images.unsplash.com/photo-1571407970349-bc81e7e96d47?w=400",
                "https://images.unsplash.com/photo-1589369056184-17b7e7df1b7b?w=400",
                "https://images.unsplash.com/photo-1606728035253-49e8a23146de?w=400",
                "https://images.unsplash.com/photo-1571066811602-716837d681de?w=400",
                "https://images.unsplash.com/photo-1595854341625-f33ee10dbf94?w=400",
                "https://images.unsplash.com/photo-1571407970371-ae4c68e3c6e8?w=400",
                "https://images.unsplash.com/photo-1607013228675-98e4ad51e1a6?w=400",
                "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400"
            ],
            "burger": [
                "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400",
                "https://images.unsplash.com/photo-1571091718767-18b5b1457add?w=400",
                "https://images.unsplash.com/photo-1594212699903-ec8a3eca50f5?w=400",
                "https://images.unsplash.com/photo-1553979459-d2229ba7433a?w=400",
                "https://images.unsplash.com/photo-1572802419224-296b0aeee0d9?w=400",
                "https://images.unsplash.com/photo-1586816001966-79b736744398?w=400",
                "https://images.unsplash.com/photo-1561758033-d89a9ad46330?w=400",
                "https://images.unsplash.com/photo-1606728035253-8b61cf15b321?w=400",
                "https://images.unsplash.com/photo-1571091655789-405eb7a3a3a8?w=400",
                "https://images.unsplash.com/photo-1596662951482-0c79e7f8bd76?w=400"
            ],
            "biryani": [
                "https://images.unsplash.com/photo-1631452180519-c014fe946bc7?w=400",
                "https://images.unsplash.com/photo-1563379091339-03246963d7d6?w=400",
                "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=400",
                "https://images.unsplash.com/photo-1599043513900-ed6ef64c6d80?w=400",
                "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=400",
                "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400",
                "https://images.unsplash.com/photo-1563379091339-03246963d7d6?w=400",
                "https://images.unsplash.com/photo-1631452181041-ce84b518ccf8?w=400",
                "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400",
                "https://images.unsplash.com/photo-1631452180629-7d1f5b7a0fa1?w=400"
            ],
            "dosa": [
                "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?w=400",
                "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?w=400",
                "https://images.unsplash.com/photo-1589301825619-0b0c8c0a3d6e?w=400",
                "https://images.unsplash.com/photo-1631452180519-c014fe946bc7?w=400",
                "https://images.unsplash.com/photo-1585937421642-6e5f6b8e5a1a?w=400",
                "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?w=400",
                "https://images.unsplash.com/photo-1589301825619-0b0c8c0a3d6e?w=400",
                "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400",
                "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?w=400",
                "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?w=400"
            ],
            "idly": [
                "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400",
                "https://images.unsplash.com/photo-1589301825619-0b0c8c0a3d6e?w=400",
                "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?w=400",
                "https://images.unsplash.com/photo-1631452180519-c014fe946bc7?w=400",
                "https://images.unsplash.com/photo-1585937421642-6e5f6b8e5a1a?w=400",
                "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?w=400",
                "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400",
                "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?w=400",
                "https://images.unsplash.com/photo-1631452180629-7d1f5b7a0fa1?w=400",
                "https://images.unsplash.com/photo-1589301825619-0b0c8c0a3d6e?w=400"
            ]
        }
    
    def download_image(self, url, filepath):
        """Download a single image from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Verify it's a valid image
            try:
                with Image.open(filepath) as img:
                    img.verify()
                return True
            except:
                os.remove(filepath)
                return False
                
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False
    
    def download_food_category(self, food_name, max_images=20):
        """Download images for a specific food category"""
        if food_name not in self.food_urls:
            print(f"❌ No URLs available for {food_name}")
            return 0
        
        # Create category directory
        category_dir = self.base_dir / food_name
        category_dir.mkdir(exist_ok=True)
        
        urls = self.food_urls[food_name]
        downloaded = 0
        
        print(f"📥 Downloading {food_name} images...")
        
        for i, url in enumerate(urls[:max_images]):
            filename = f"{food_name}_{i+1:03d}.jpg"
            filepath = category_dir / filename
            
            if filepath.exists():
                print(f"   ⏭️  Skipping {filename} (already exists)")
                downloaded += 1
                continue
            
            print(f"   📥 Downloading {filename}...")
            if self.download_image(url, filepath):
                downloaded += 1
                print(f"   ✅ Downloaded {filename}")
            else:
                print(f"   ❌ Failed {filename}")
            
            # Be respectful to servers
            time.sleep(random.uniform(0.5, 1.5))
        
        print(f"✅ Downloaded {downloaded} images for {food_name}")
        return downloaded
    
    def download_all_categories(self, max_per_category=20):
        """Download images for all food categories"""
        print("🍽️ Downloading Food Images from Internet")
        print("=" * 50)
        
        total_downloaded = 0
        for food_name in self.food_urls.keys():
            count = self.download_food_category(food_name, max_per_category)
            total_downloaded += count
            print()
        
        print(f"🎉 Total images downloaded: {total_downloaded}")
        return total_downloaded
    
    def create_train_val_split(self, train_ratio=0.8):
        """Create train/validation split"""
        print("📁 Creating train/validation split...")
        
        train_dir = self.base_dir / "train"
        val_dir = self.base_dir / "val"
        
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        for food_name in self.food_urls.keys():
            # Create subdirectories
            (train_dir / food_name).mkdir(exist_ok=True)
            (val_dir / food_name).mkdir(exist_ok=True)
            
            # Get all images for this category
            category_dir = self.base_dir / food_name
            if not category_dir.exists():
                continue
                
            images = list(category_dir.glob("*.jpg"))
            random.shuffle(images)
            
            # Split into train/val
            split_idx = int(len(images) * train_ratio)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Move images
            for img in train_images:
                new_path = train_dir / food_name / img.name
                if not new_path.exists():
                    import shutil
                    shutil.copy2(img, new_path)
            
            for img in val_images:
                new_path = val_dir / food_name / img.name
                if not new_path.exists():
                    import shutil
                    shutil.copy2(img, new_path)
            
            print(f"   {food_name}: {len(train_images)} train, {len(val_images)} val")
        
        print("✅ Train/validation split completed")

def main():
    """Main function to download food images"""
    downloader = InternetFoodImageDownloader()
    
    # Download images
    total = downloader.download_all_categories(max_per_category=15)
    
    if total > 0:
        # Create train/val split
        downloader.create_train_val_split()
        
        print(f"\n🎯 Dataset ready for training!")
        print(f"📁 Images saved to: {downloader.base_dir}")
        print(f"🚀 Next step: Run training script")
    else:
        print("❌ No images downloaded. Please check internet connection.")

if __name__ == "__main__":
    main()
