#!/usr/bin/env python3
"""
Generate synthetic food images using PIL for training when real images are limited
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import colorsys
from pathlib import Path
import numpy as np

class SyntheticFoodGenerator:
    def __init__(self, output_dir="data/synthetic_food"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Food color schemes
        self.food_colors = {
            "pizza": [(255, 200, 100), (255, 150, 50), (200, 100, 50), (255, 100, 100)],
            "burger": [(139, 69, 19), (255, 200, 100), (34, 139, 34), (255, 99, 71)],
            "biryani": [(255, 215, 0), (255, 140, 0), (139, 69, 19), (255, 255, 255)],
            "dosa": [(255, 228, 181), (255, 200, 120), (205, 133, 63), (255, 255, 255)],
            "idly": [(255, 255, 255), (245, 245, 220), (255, 250, 240), (248, 248, 255)]
        }
    
    def generate_pizza(self, size=(224, 224)):
        """Generate a synthetic pizza image"""
        img = Image.new('RGB', size, color=(50, 50, 50))  # Dark background
        draw = ImageDraw.Draw(img)
        
        # Pizza base (circle)
        center = (size[0]//2, size[1]//2)
        radius = min(size) // 3
        
        # Base dough
        draw.ellipse([center[0]-radius, center[1]-radius, 
                     center[0]+radius, center[1]+radius], 
                    fill=(255, 200, 100), outline=(200, 150, 50), width=3)
        
        # Add toppings (random circles)
        for _ in range(random.randint(8, 15)):
            x = random.randint(center[0]-radius//2, center[0]+radius//2)
            y = random.randint(center[1]-radius//2, center[1]+radius//2)
            r = random.randint(8, 20)
            color = random.choice([(255, 100, 100), (255, 150, 50), (100, 255, 100)])
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
        
        return img
    
    def generate_burger(self, size=(224, 224)):
        """Generate a synthetic burger image"""
        img = Image.new('RGB', size, color=(100, 100, 100))
        draw = ImageDraw.Draw(img)
        
        center_x = size[0] // 2
        
        # Burger layers (rectangles with rounded corners)
        layers = [
            (center_x-60, 80, center_x+60, 100, (139, 69, 19)),   # Top bun
            (center_x-70, 100, center_x+70, 115, (34, 139, 34)),  # Lettuce
            (center_x-65, 115, center_x+65, 135, (139, 69, 19)),  # Patty
            (center_x-60, 135, center_x+60, 150, (255, 255, 0)),  # Cheese
            (center_x-60, 150, center_x+60, 170, (139, 69, 19))   # Bottom bun
        ]
        
        for layer in layers:
            draw.rectangle(layer[:4], fill=layer[4])
        
        return img
    
    def generate_biryani(self, size=(224, 224)):
        """Generate a synthetic biryani image"""
        img = Image.new('RGB', size, color=(139, 69, 19))  # Bowl color
        draw = ImageDraw.Draw(img)
        
        # Bowl outline
        draw.ellipse([30, 50, size[0]-30, size[1]-30], 
                    fill=(255, 215, 0), outline=(200, 150, 0), width=5)
        
        # Rice grains (small lines)
        for _ in range(100):
            x = random.randint(50, size[0]-50)
            y = random.randint(70, size[1]-50)
            length = random.randint(3, 8)
            angle = random.randint(0, 360)
            color = random.choice([(255, 215, 0), (255, 255, 255), (255, 140, 0)])
            
            # Draw small line for rice grain
            end_x = x + length * np.cos(np.radians(angle))
            end_y = y + length * np.sin(np.radians(angle))
            draw.line([(x, y), (end_x, end_y)], fill=color, width=2)
        
        return img
    
    def generate_dosa(self, size=(224, 224)):
        """Generate a synthetic dosa image"""
        img = Image.new('RGB', size, color=(50, 50, 50))
        draw = ImageDraw.Draw(img)
        
        # Dosa (circular/oval shape)
        draw.ellipse([20, 40, size[0]-20, size[1]-40], 
                    fill=(255, 228, 181), outline=(200, 160, 100), width=3)
        
        # Add texture lines
        for i in range(5, size[0]-5, 20):
            draw.line([(i, 50), (i+10, size[1]-50)], 
                     fill=(200, 160, 100), width=1)
        
        # Coconut chutney (small circle)
        draw.ellipse([size[0]-60, size[1]-60, size[0]-20, size[1]-20], 
                    fill=(255, 255, 255))
        
        return img
    
    def generate_idly(self, size=(224, 224)):
        """Generate a synthetic idly image"""
        img = Image.new('RGB', size, color=(100, 100, 100))
        draw = ImageDraw.Draw(img)
        
        # Multiple idly pieces (circles)
        positions = [(80, 80), (140, 80), (80, 140), (140, 140)]
        
        for pos in positions:
            # Main idly
            draw.ellipse([pos[0]-25, pos[1]-15, pos[0]+25, pos[1]+15], 
                        fill=(255, 255, 255), outline=(200, 200, 200), width=2)
            
            # Small highlight
            draw.ellipse([pos[0]-15, pos[1]-8, pos[0]-5, pos[1]-3], 
                        fill=(255, 255, 240))
        
        return img
    
    def generate_food_images(self, food_type, count=20):
        """Generate multiple images for a food type"""
        print(f"🎨 Generating {count} synthetic {food_type} images...")
        
        # Create directory
        food_dir = self.output_dir / food_type
        food_dir.mkdir(exist_ok=True)
        
        generator_map = {
            "pizza": self.generate_pizza,
            "burger": self.generate_burger,
            "biryani": self.generate_biryani,
            "dosa": self.generate_dosa,
            "idly": self.generate_idly
        }
        
        if food_type not in generator_map:
            print(f"❌ No generator for {food_type}")
            return 0
        
        generator = generator_map[food_type]
        generated = 0
        
        for i in range(count):
            try:
                # Generate image
                img = generator()
                
                # Add some random noise/variations
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.5)))
                
                # Save image
                filename = f"synthetic_{food_type}_{i+1:03d}.jpg"
                filepath = food_dir / filename
                img.save(filepath, "JPEG", quality=90)
                generated += 1
                
            except Exception as e:
                print(f"❌ Failed to generate {food_type} image {i+1}: {e}")
        
        print(f"✅ Generated {generated} {food_type} images")
        return generated
    
    def generate_all_foods(self, count_per_food=30):
        """Generate synthetic images for all food types"""
        print("🎨 Generating Synthetic Food Dataset")
        print("=" * 50)
        
        food_types = ["pizza", "burger", "biryani", "dosa", "idly"]
        total_generated = 0
        
        for food_type in food_types:
            count = self.generate_food_images(food_type, count_per_food)
            total_generated += count
            print()
        
        print(f"🎉 Total synthetic images generated: {total_generated}")
        
        # Create train/val split
        self.create_train_val_split()
        
        return total_generated
    
    def create_train_val_split(self):
        """Create train/validation split for synthetic data"""
        print("📁 Creating train/validation split for synthetic data...")
        
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        food_types = ["pizza", "burger", "biryani", "dosa", "idly"]
        
        for food_type in food_types:
            # Create subdirectories
            (train_dir / food_type).mkdir(exist_ok=True)
            (val_dir / food_type).mkdir(exist_ok=True)
            
            # Get all images
            food_dir = self.output_dir / food_type
            if not food_dir.exists():
                continue
            
            images = list(food_dir.glob("*.jpg"))
            random.shuffle(images)
            
            # 80/20 split
            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Copy images
            import shutil
            for img in train_images:
                new_path = train_dir / food_type / img.name
                if not new_path.exists():
                    shutil.copy2(img, new_path)
            
            for img in val_images:
                new_path = val_dir / food_type / img.name
                if not new_path.exists():
                    shutil.copy2(img, new_path)
            
            print(f"   {food_type}: {len(train_images)} train, {len(val_images)} val")

def main():
    """Generate synthetic food dataset"""
    generator = SyntheticFoodGenerator()
    total = generator.generate_all_foods(count_per_food=25)
    
    if total > 0:
        print(f"\n🎯 Synthetic dataset ready!")
        print(f"📁 Images saved to: {generator.output_dir}")
        print(f"🚀 Ready for training!")

if __name__ == "__main__":
    main()
