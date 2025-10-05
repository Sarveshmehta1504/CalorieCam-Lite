#!/usr/bin/env python3
"""
Process the downloaded Kaggle dataset and create proper calorie map
"""
import pandas as pd
import re
import os

def process_kaggle_food_data():
    """Process the Kaggle food and calories dataset"""
    print("🔄 Processing Kaggle Food and Calories dataset...")
    
    # Read the dataset
    df = pd.read_csv('data/kaggle_food_calories.csv')
    print(f"📊 Loaded {len(df)} food items")
    
    # Process the data
    processed_data = []
    
    for _, row in df.iterrows():
        food_name = row['Food'].strip().lower()
        serving = row['Serving'].strip()
        calories_str = row['Calories'].strip()
        
        # Extract numeric calories (remove 'cal' text)
        calories_match = re.search(r'(\d+)', calories_str)
        if calories_match:
            calories = int(calories_match.group(1))
            processed_data.append({
                'label': food_name,
                'avg_calories_kcal': calories,
                'portion_notes': serving
            })
    
    # Create DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    # Remove duplicates
    processed_df = processed_df.drop_duplicates(subset=['label'], keep='first')
    
    print(f"✅ Processed {len(processed_df)} unique food items")
    
    # Save enhanced calorie map
    processed_df.to_csv('data/enhanced_calorie_map.csv', index=False)
    print("✅ Saved enhanced_calorie_map.csv")
    
    # Display sample
    print("\n🍕 Sample from processed data:")
    print(processed_df.head(10).to_string())
    
    # Merge with existing calorie map
    merge_calorie_maps()
    
    return processed_df

def merge_calorie_maps():
    """Merge the new data with existing calorie map"""
    try:
        # Read existing and new data
        existing_df = pd.read_csv('data/calorie_map.csv')
        new_df = pd.read_csv('data/enhanced_calorie_map.csv')
        
        print(f"\n📄 Existing map: {len(existing_df)} items")
        print(f"📄 New data: {len(new_df)} items")
        
        # Backup original
        existing_df.to_csv('data/calorie_map_backup.csv', index=False)
        print("💾 Created backup: calorie_map_backup.csv")
        
        # Combine data - existing takes priority for duplicates
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['label'], keep='first')
        
        # Sort by label
        combined_df = combined_df.sort_values('label').reset_index(drop=True)
        
        # Save merged data
        combined_df.to_csv('data/calorie_map.csv', index=False)
        
        print(f"✅ Updated calorie_map.csv with {len(combined_df)} total items")
        
        # Show some additions
        new_additions = combined_df[~combined_df['label'].isin(existing_df['label'])]
        if len(new_additions) > 0:
            print(f"\n🆕 Added {len(new_additions)} new food items:")
            print(new_additions.head(10)[['label', 'avg_calories_kcal']].to_string())
        
    except Exception as e:
        print(f"❌ Error merging calorie maps: {e}")

def main():
    """Main processing function"""
    print("🍽️ CalorieCam Lite - Process Kaggle Dataset")
    print("=" * 50)
    
    if not os.path.exists('data/kaggle_food_calories.csv'):
        print("❌ Kaggle dataset not found. Please run download_kaggle_dataset.py first.")
        return
    
    processed_df = process_kaggle_food_data()
    
    print(f"\n🎉 Processing completed!")
    print(f"📊 Total unique food items processed: {len(processed_df)}")
    print("\n📁 Files updated:")
    print("   - data/enhanced_calorie_map.csv (processed Kaggle data)")
    print("   - data/calorie_map.csv (merged with existing data)")
    print("   - data/calorie_map_backup.csv (backup of original)")

if __name__ == "__main__":
    main()
