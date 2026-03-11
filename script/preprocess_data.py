import os
import shutil
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_directory_structure():
    """
    Create the required directory structure for training
    """
    directories = [
        'data/train/Normal',
        'data/train/LSIL',
        'data/train/HSIL',
        'data/train/SCC',
        'data/validation/Normal',
        'data/validation/LSIL',
        'data/validation/HSIL',
        'data/validation/SCC',
        'data/test/Normal',
        'data/test/LSIL',
        'data/test/HSIL',
        'data/test/SCC'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directory structure created successfully!")

def validate_and_convert_images(source_dir, target_size=(224, 224)):
    """
    Validate images and convert them to standard format
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    corrupted_images = []
    processed_count = 0
    
    for root, dirs, files in os.walk(source_dir):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in valid_extensions:
                try:
                    # Open and validate image
                    img = Image.open(file_path)
                    img.verify()  # Verify it's not corrupted
                    
                    # Reopen (verify closes the file)
                    img = Image.open(file_path)
                    
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Optional: Resize to standard size
                    # img = img.resize(target_size, Image.LANCZOS)
                    
                    # Save back
                    img.save(file_path, 'JPEG', quality=95)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"\n❌ Corrupted image: {file_path}")
                    print(f"   Error: {str(e)}")
                    corrupted_images.append(file_path)
    
    print(f"\n✅ Processed {processed_count} images")
    if corrupted_images:
        print(f"⚠️  Found {len(corrupted_images)} corrupted images")
        print("Corrupted images list:")
        for img in corrupted_images:
            print(f"  - {img}")
    
    return corrupted_images

def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.001, \
        "Ratios must sum to 1.0"
    
    classes = ['Normal', 'LSIL', 'HSIL', 'SCC']
    
    for class_name in classes:
        source_class_dir = os.path.join(source_dir, class_name)
        
        if not os.path.exists(source_class_dir):
            print(f"⚠️  Warning: {source_class_dir} does not exist, skipping...")
            continue
        
        # Get all image files
        images = [f for f in os.listdir(source_class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(images) == 0:
            print(f"⚠️  No images found in {source_class_dir}")
            continue
        
        # First split: train and temp (validation + test)
        train_files, temp_files = train_test_split(
            images, 
            test_size=(val_ratio + test_ratio),
            random_state=42
        )
        
        # Second split: validation and test
        val_files, test_files = train_test_split(
            temp_files,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=42
        )
        
        # Copy files to respective directories
        print(f"\n📁 Processing class: {class_name}")
        print(f"   Train: {len(train_files)} images")
        print(f"   Validation: {len(val_files)} images")
        print(f"   Test: {len(test_files)} images")
        
        # Copy train files
        for file in tqdm(train_files, desc="Copying train files"):
            src = os.path.join(source_class_dir, file)
            dst = os.path.join('data/train', class_name, file)
            shutil.copy2(src, dst)
        
        # Copy validation files
        for file in tqdm(val_files, desc="Copying validation files"):
            src = os.path.join(source_class_dir, file)
            dst = os.path.join('data/validation', class_name, file)
            shutil.copy2(src, dst)
        
        # Copy test files
        for file in tqdm(test_files, desc="Copying test files"):
            src = os.path.join(source_class_dir, file)
            dst = os.path.join('data/test', class_name, file)
            shutil.copy2(src, dst)
    
    print("\n✅ Dataset split completed!")

def analyze_dataset(data_dir='data'):
    """
    Analyze and display dataset statistics
    """
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    splits = ['train', 'validation', 'test']
    classes = ['Normal', 'LSIL', 'HSIL', 'SCC']
    
    total_images = 0
    
    for split in splits:
        print(f"\n📊 {split.upper()} SET:")
        split_total = 0
        
        for class_name in classes:
            class_dir = os.path.join(data_dir, split, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                print(f"   {class_name:30} : {count:5} images")
                split_total += count
            else:
                print(f"   {class_name:30} : Directory not found")
        
        print(f"   {'-'*40}")
        print(f"   {'Total':30} : {split_total:5} images")
        total_images += split_total
    
    print(f"\n{'='*60}")
    print(f"TOTAL DATASET: {total_images} images")
    print(f"{'='*60}\n")

def check_class_balance(data_dir='data/train'):
    """
    Check class balance and suggest weights if imbalanced
    """
    classes = ['Normal', 'LSIL', 'HSIL', 'SCC']
    class_counts = {}
    
    print("\n" + "="*60)
    print("CLASS BALANCE ANALYSIS")
    print("="*60)
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            class_counts[class_name] = count
    
    total = sum(class_counts.values())
    
    print("\nClass Distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"   {class_name:30} : {count:5} images ({percentage:5.2f}%)")
    
    # Calculate class weights for imbalanced datasets
    if total > 0:
        max_count = max(class_counts.values())
        class_weights = {i: max_count / count 
                        for i, (class_name, count) in enumerate(class_counts.items())}
        
        print("\n💡 Suggested Class Weights (for imbalanced dataset):")
        for i, (class_name, weight) in enumerate(zip(class_counts.keys(), class_weights.values())):
            print(f"   {i}: {weight:.2f}  ({class_name})")
        
        print("\nAdd to model.fit():")
        print(f"   class_weight={class_weights}")
    
    print("="*60 + "\n")

def main():
    """
    Main preprocessing pipeline
    """
    print("\n" + "="*60)
    print("CERVICAL CANCER DATASET PREPROCESSING")
    print("="*60 + "\n")
    
    # Step 1: Create directory structure
    print("Step 1: Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Validate images (if you have a raw dataset)
    # Uncomment and adjust path if needed
    # print("\nStep 2: Validating images...")
    # validate_and_convert_images('raw_data')
    
    # Step 3: Split dataset
    # Uncomment and adjust path if you have unsplit data
    print("\nStep 3: Splitting dataset...")
    print("⚠️  Make sure your source data is in the correct format:")
    print("   source_data/")
    print("   ├── Normal/")
    print("   ├── LSIL/")
    print("   ├── HSIL/")
    print("   └── SCC/")
    
    source_data_path = input("\nEnter path to source data (or press Enter to skip): ").strip()
    
    if source_data_path and os.path.exists(source_data_path):
        split_dataset(source_data_path)
    else:
        print("⏭️  Skipping dataset split...")
    
    # Step 4: Analyze dataset
    print("\nStep 4: Analyzing dataset...")
    analyze_dataset()
    
    # Step 5: Check class balance
    check_class_balance()
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the dataset statistics above")
    print("2. Adjust class weights if dataset is imbalanced")
    print("3. Run train_model.py to start training")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()