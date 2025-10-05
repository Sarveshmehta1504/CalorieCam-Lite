#!/usr/bin/env python3
"""
Complete dataset setup for CalorieCam Lite
Downloads Kaggle dataset and sets up training data
"""
import os
import sys
import pandas as pd
import kagglehub
from pathlib import Path

def setup_kaggle_credentials():
    """Check and guide user to set up Kaggle credentials"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("❌ Kaggle API credentials not found!")
        print("\n🔧 To download datasets from Kaggle, you need to set up API credentials:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Click 'Create New API Token'")
        print("   3. This will download kaggle.json")
        print("   4. Move it to ~/.kaggle/kaggle.json")
        print("   5. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("\n💡 After setting up credentials, run this script again.")
        return False
    
    print("✅ Kaggle credentials found")
    return True

def download_and_process_dataset():
    """Download and process the Kaggle food and calories dataset"""
    print("🍽️ CalorieCam Lite - Complete Dataset Setup")
    print("=" * 60)
    
    # Check Kaggle credentials
    if not setup_kaggle_credentials():
        return False
    
    try:
        # Download dataset
        print("\n🔽 Downloading Food and Calories dataset from Kaggle...")
        path = kagglehub.dataset_download("vaishnavivenkatesan/food-and-their-calories")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Find the CSV file
        csv_files = list(Path(path).glob("*.csv"))
        if not csv_files:
            print("❌ No CSV files found in dataset")
            return False
        
        # Read and process the dataset
        dataset_file = csv_files[0]
        df = pd.read_csv(dataset_file)
        print(f"📊 Loaded {len(df)} food items from {dataset_file.name}")
        
        # Save original dataset
        original_output = "data/kaggle_food_calories.csv"
        os.makedirs("data", exist_ok=True)
        df.to_csv(original_output, index=False)
        print(f"✅ Saved original dataset to: {original_output}")
        
        # Process the data to create calorie map
        processed_data = []
        import re
        
        for _, row in df.iterrows():
            food_name = str(row['Food']).strip().lower()
            serving = str(row['Serving']).strip()
            calories_str = str(row['Calories']).strip()
            
            # Extract numeric calories
            calories_match = re.search(r'(\d+)', calories_str)
            if calories_match:
                calories = int(calories_match.group(1))
                processed_data.append({
                    'label': food_name,
                    'avg_calories_kcal': calories,
                    'portion_notes': serving
                })
        
        # Create processed DataFrame
        processed_df = pd.DataFrame(processed_data)
        processed_df = processed_df.drop_duplicates(subset=['label'], keep='first')
        processed_df = processed_df.sort_values('label').reset_index(drop=True)
        
        print(f"✅ Processed {len(processed_df)} unique food items")
        
        # Save enhanced calorie map
        enhanced_output = "data/enhanced_calorie_map.csv"
        processed_df.to_csv(enhanced_output, index=False)
        print(f"✅ Saved enhanced calorie map to: {enhanced_output}")
        
        # Merge with existing calorie map if it exists
        existing_calorie_map = "data/calorie_map.csv"
        if os.path.exists(existing_calorie_map):
            existing_df = pd.read_csv(existing_calorie_map)
            print(f"📄 Found existing calorie map with {len(existing_df)} items")
            
            # Backup original
            backup_file = "data/calorie_map_backup.csv"
            existing_df.to_csv(backup_file, index=False)
            print(f"💾 Created backup: {backup_file}")
            
            # Merge data (existing takes priority)
            combined_df = pd.concat([existing_df, processed_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['label'], keep='first')
            combined_df = combined_df.sort_values('label').reset_index(drop=True)
            
            # Save merged data
            combined_df.to_csv(existing_calorie_map, index=False)
            print(f"✅ Updated calorie_map.csv with {len(combined_df)} total items")
            
            # Show statistics
            new_items = len(combined_df) - len(existing_df)
            if new_items > 0:
                print(f"🆕 Added {new_items} new food items to the database")
        else:
            # If no existing map, use the processed data as the main map
            processed_df.to_csv(existing_calorie_map, index=False)
            print(f"✅ Created new calorie_map.csv with {len(processed_df)} items")
        
        # Display sample data
        print("\n🍕 Sample from updated calorie database:")
        final_df = pd.read_csv(existing_calorie_map)
        sample_data = final_df.sample(min(10, len(final_df)))
        for _, row in sample_data.iterrows():
            print(f"   • {row['label'].title()}: {row['avg_calories_kcal']:.0f} kcal ({row['portion_notes']})")
        
        print(f"\n🎉 Dataset setup completed successfully!")
        print(f"📊 Total food items in database: {len(final_df)}")
        print("\n📁 Files created/updated:")
        print("   • data/kaggle_food_calories.csv (original Kaggle dataset)")
        print("   • data/enhanced_calorie_map.csv (processed Kaggle data)")
        print("   • data/calorie_map.csv (main calorie database)")
        print("   • data/calorie_map_backup.csv (backup of previous version)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = download_and_process_dataset()
    
    if success:
        print("\n🚀 Next steps:")
        print("   1. Your CalorieCam Lite now has a comprehensive food database!")
        print("   2. Train a model: python src/train.py")
        print("   3. Test the app: streamlit run app/streamlit_app.py")
        print("   4. Upload food images and get calorie predictions!")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
