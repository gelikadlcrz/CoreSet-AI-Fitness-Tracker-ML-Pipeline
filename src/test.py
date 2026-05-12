import os
from collections import Counter

# Using the exact absolute path from your previous test
DATA_DIR = r"C:\Users\dcand\Downloads\CoreSet-AI-Fitness-Tracker-ML-Pipeline\data\final_data"

def count_dataset_files():
    print("=" * 40)
    print(" 📊 CoreSet Dataset Counter")
    print("=" * 40)
    print(f"Scanning: {DATA_DIR}\n")
    
    if not os.path.exists(DATA_DIR):
        print("❌ Error: Directory not found. Double check the path!")
        return

    category_counts = Counter()
    total_files = 0

    # Walk through all folders and subfolders
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            # Only count the final .npz archives
            if file.endswith(".npz"):
                # The parent folder name is the exercise category
                category = os.path.basename(root)
                category_counts[category] += 1
                total_files += 1

    # Print the breakdown per exercise
    for category, count in sorted(category_counts.items()):
        print(f"  ▶ {category.ljust(15)} : {count} files")
        
    print("-" * 40)
    print(f"  TOTAL .npz FILES  : {total_files}")
    print("-" * 40)
    
    # Quick health check against your 1,000 file target
    expected_total = 1000
    if total_files < expected_total:
        missing = expected_total - total_files
        print(f"\n⚠️ Missing {missing} files to reach your {expected_total} target.")
        print("Check your raw data conversion pipeline to see where it stopped!")
    elif total_files == expected_total:
        print(f"\n✅ Target hit! All {expected_total} files are present and accounted for.")

if __name__ == "__main__":
    count_dataset_files()