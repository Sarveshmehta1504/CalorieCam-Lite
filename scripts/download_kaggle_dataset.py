#!/usr/bin/env python3
"""
Download and process Kaggle Food and Calories dataset
"""
import kagglehub
import pandas as pd
import os
import shutil
from pathlib import Path

def download_kaggle_dataset():
    """Download the food and calories dataset from Kaggle"""
    print("🔽 Downloading Food and Calories dataset from Kaggle...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("vaishnavivenkatesan/food-and-their-calories")
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Make sure you have Kaggle API credentials set up:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Click 'Create New API Token'")
        print("   3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return None

def process_dataset(dataset_path, output_dir="data"):
    """Process the downloaded dataset"""
    if not dataset_path or not os.path.exists(dataset_path):
        print("❌ Dataset path not found")
        return False
    
    print(f"📁 Processing dataset from: {dataset_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find CSV files in the dataset
    csv_files = list(Path(dataset_path).glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in dataset")
        return False
    
    print(f"📄 Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        print(f"   - {csv_file.name}")
    
    # Process the main CSV file
    main_csv = csv_files[0]  # Use the first CSV file found
    
    try:
        # Read the dataset
        df = pd.read_csv(main_csv)
        print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Display first few rows
        print("\n📋 First 5 rows:")
        print(df.head())
        
        # Copy the original file to our data directory
        output_file = os.path.join(output_dir, "kaggle_food_calories.csv")
        shutil.copy2(main_csv, output_file)
        print(f"✅ Copied dataset to: {output_file}")
        
        # Try to create a standardized calorie map
        create_calorie_map(df, output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return False

def create_calorie_map(df, output_dir):
    """Create a standardized calorie map from the dataset"""
    try:
        # Try to identify food name and calorie columns
        columns = df.columns.str.lower()
        
        # Common column name patterns
        food_cols = [col for col in df.columns if any(term in col.lower() for term in ['food', 'name', 'item', 'dish'])]
        calorie_cols = [col for col in df.columns if any(term in col.lower() for term in ['calorie', 'kcal', 'energy'])]
        
        if not food_cols or not calorie_cols:
            print("⚠️  Could not identify food and calorie columns automatically")
            print(f"   Available columns: {list(df.columns)}")
            return
        
        food_col = food_cols[0]
        calorie_col = calorie_cols[0]
        
        print(f"🍽️  Using food column: '{food_col}'")
        print(f"🔥 Using calorie column: '{calorie_col}'")
        
        # Create clean calorie map
        calorie_map = df[[food_col, calorie_col]].copy()
        calorie_map.columns = ['label', 'avg_calories_kcal']
        
        # Clean data
        calorie_map = calorie_map.dropna()
        calorie_map['label'] = calorie_map['label'].str.lower().str.strip()
        calorie_map['avg_calories_kcal'] = pd.to_numeric(calorie_map['avg_calories_kcal'], errors='coerce')
        calorie_map = calorie_map.dropna()
        
        # Remove duplicates, keep first occurrence
        calorie_map = calorie_map.drop_duplicates(subset=['label'], keep='first')
        
        # Add portion notes column
        calorie_map['portion_notes'] = "1 serving (average)"
        
        # Save the enhanced calorie map
        enhanced_file = os.path.join(output_dir, "enhanced_calorie_map.csv")
        calorie_map.to_csv(enhanced_file, index=False)
        print(f"✅ Created enhanced calorie map: {enhanced_file}")
        print(f"📊 Contains {len(calorie_map)} unique food items")
        
        # Display sample of enhanced data
        print("\n🍕 Sample from enhanced calorie map:")
        print(calorie_map.head(10))
        
        # Optionally merge with existing calorie map
        merge_with_existing(calorie_map, output_dir)
        
    except Exception as e:
        print(f"❌ Error creating calorie map: {e}")

def merge_with_existing(new_data, output_dir):
    """Merge new calorie data with existing calorie_map.csv"""
    existing_file = os.path.join(output_dir, "calorie_map.csv")
    
    if os.path.exists(existing_file):
        try:
            existing_df = pd.read_csv(existing_file)
            print(f"📄 Found existing calorie map with {len(existing_df)} items")
            
            # Merge data, keeping existing values for duplicates
            merged_df = pd.concat([existing_df, new_data], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=['label'], keep='first')
            
            # Backup original
            backup_file = os.path.join(output_dir, "calorie_map_backup.csv")
            shutil.copy2(existing_file, backup_file)
            print(f"💾 Backed up original to: {backup_file}")
            
            # Save merged data
            merged_df.to_csv(existing_file, index=False)
            print(f"✅ Updated calorie map with {len(merged_df)} total items")
            
        except Exception as e:
            print(f"⚠️  Could not merge with existing calorie map: {e}")

def main():
    """Main function to download and process the dataset"""
    print("🍽️ CalorieCam Lite - Kaggle Dataset Download")
    print("=" * 50)
    
    # Download dataset
    dataset_path = download_kaggle_dataset()
    
    if dataset_path:
        # Process dataset
        success = process_dataset(dataset_path)
        
        if success:
            print("\n🎉 Dataset download and processing completed successfully!")
            print("\n📁 Files created:")
            print("   - data/kaggle_food_calories.csv (original dataset)")
            print("   - data/enhanced_calorie_map.csv (processed calorie map)")
            print("   - data/calorie_map.csv (updated with new data)")
            print("\n🚀 You can now use the enhanced dataset for training!")
        else:
            print("\n❌ Dataset processing failed")
    else:
        print("\n❌ Dataset download failed")

if __name__ == "__main__":
    main()
