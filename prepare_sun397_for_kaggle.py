"""
SUN397 Dataset Preparation for Kaggle Competition (Final Evaluation)
=====================================================================

Creates Kaggle competition from SUN397 (Scene Understanding):
- All 397 scene classes
- ~108k images
- Resize to 96x96 during processing (memory efficient)

Usage:
    python prepare_sun397_for_kaggle.py \
        --output_dir ./kaggle_data_sun397 \
        --resolution 96
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import tarfile
from PIL import Image
from tqdm import tqdm
import argparse


def download_sun397(download_dir):
    """
    Download SUN397 dataset using HuggingFace datasets
    
    This is much more reliable than the old URL!
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    print("="*60)
    print("Downloading SUN397 from HuggingFace...")
    print("This will download ~37 GB (original resolution)")
    print("We'll resize to 96px during processing to save space")
    print("="*60)
    
    # Load SUN397 from HuggingFace
    print("\nLoading dataset (this may take a while)...")
    dataset = load_dataset("tanganke/sun397", cache_dir=str(download_dir))
    
    print(f"\nDataset loaded!")
    print(f"  Train: {len(dataset['train'])} images")
    if 'test' in dataset:
        print(f"  Test: {len(dataset['test'])} images")
    
    return dataset


def load_sun397_structure(dataset):
    """
    Load SUN397 dataset structure from HuggingFace dataset
    
    HuggingFace format: {'image': PIL.Image, 'label': int}
    """
    print("\n" + "="*60)
    print("Processing SUN397 structure...")
    print("="*60)
    
    # Get train split
    train_data = dataset['train']
    
    # Get label names
    label_names = train_data.features['label'].names
    
    print(f"\nTotal classes: {len(label_names)}")
    print(f"Total images: {len(train_data)}")
    
    print(f"\nSample classes:")
    for idx, name in enumerate(label_names[:10]):
        print(f"  {idx}: {name}")
    print("  ...")
    
    # Count images per class
    class_counts = {}
    for item in train_data:
        label = item['label']
        class_counts[label] = class_counts.get(label, 0) + 1
    
    class_info = [
        {
            'class_id': idx,
            'class_name': name,
            'n_images': class_counts.get(idx, 0)
        }
        for idx, name in enumerate(label_names)
    ]
    
    return class_info, train_data


def create_kaggle_dataset(
    hf_dataset,
    output_dir,
    class_info,
    resolution=96,
    seed=42
):
    """
    Create Kaggle competition dataset from SUN397 HuggingFace dataset
    Process images at 96px to keep memory low!
    
    Args:
        hf_dataset: HuggingFace dataset (train split)
        output_dir: Output directory
        class_info: List of class information dicts
        resolution: Target resolution (96, 224, etc.)
        seed: Random seed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(seed)
    
    print("\n" + "="*60)
    print(f"Creating SUN397 Kaggle dataset")
    print(f"  Classes: {len(class_info)}")
    print(f"  Images: {len(hf_dataset)}")
    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  (Resizing during save to keep memory low)")
    print("="*60)
    
    # Collect all images with labels
    print("\nCollecting all images...")
    all_data = []
    
    for idx in tqdm(range(len(hf_dataset)), desc="Loading images"):
        item = hf_dataset[idx]
        all_data.append({
            'hf_index': idx,
            'image': item['image'],  # PIL Image
            'class_id': item['label'],
            'class_name': class_info[item['label']]['class_name']
        })
    
    print(f"Total images collected: {len(all_data)}")
    
    # Shuffle and split
    np.random.shuffle(all_data)
    
    n_total = len(all_data)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} images")
    print(f"  Val: {len(val_data)} images")
    print(f"  Test: {len(test_data)} images")
    
    # Save images (resize during save to keep memory low!)
    def save_split(split_data, split_name):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        print(f"\nSaving {split_name} images (resize to {resolution}x{resolution})...")
        for idx, item in enumerate(tqdm(split_data, desc=f"{split_name}")):
            try:
                # Get PIL image from HuggingFace dataset
                img = item['image'].convert('RGB')
                
                # Resize to target resolution (memory efficient!)
                img = img.resize((resolution, resolution), Image.BILINEAR)
                
                # Save with new filename
                filename = f"{idx:06d}_class{item['class_id']:03d}.jpg"
                img.save(split_dir / filename, quality=85)
                
                metadata.append({
                    'filename': filename,
                    'class_id': item['class_id'],
                    'class_name': item['class_name']
                })
                
                # Clear memory
                del img
                
            except Exception as e:
                print(f"\nWarning: Failed to process image {idx}: {e}")
                continue
        
        return pd.DataFrame(metadata)
    
    # Save all splits
    train_df = save_split(train_data, 'train')
    val_df = save_split(val_data, 'val')
    test_df = save_split(test_data, 'test')
    
    # Save CSVs
    train_df.to_csv(output_dir / 'train_labels.csv', index=False)
    val_df.to_csv(output_dir / 'val_labels.csv', index=False)
    
    # Test: save internal labels and public version
    test_df.to_csv(output_dir / 'test_labels_INTERNAL.csv', index=False)
    test_df[['filename']].to_csv(output_dir / 'test_images.csv', index=False)
    
    # Create sample submission (solution.csv removed - for TAs only!)
    create_sample_submission(test_df, output_dir)
    
    # Create class mapping file
    class_mapping_df = pd.DataFrame(class_info)
    class_mapping_df.to_csv(output_dir / 'class_mapping.csv', index=False)
    
    # Create README
    create_readme(output_dir, len(class_info), resolution)
    
    print(f"\n{'='*60}")
    print("Dataset creation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    
    return train_df, val_df, test_df


def create_sample_submission(test_df, output_dir):
    """Create sample_submission.csv"""
    output_dir = Path(output_dir)
    
    sample_submission = pd.DataFrame({
        'id': test_df['filename'],
        'class_id': 0
    })
    
    sample_submission.to_csv(output_dir / 'sample_submission.csv', index=False)
    print(f"\nSample submission created")


def create_readme(output_dir, num_classes, resolution):
    """Create README for dataset"""
    readme = f"""# SUN397 Scene Recognition Dataset (Final Evaluation)

## Dataset Info

- **Source**: SUN397 (Scene UNderstanding)
- **Classes**: {num_classes} scene categories
- **Resolution**: {resolution}x{resolution}
- **Task**: Scene recognition (indoor/outdoor environments)

## Files

- `train/` - Training images with labels
- `val/` - Validation images with labels
- `test/` - Test images (no labels)
- `train_labels.csv` - Training labels
- `val_labels.csv` - Validation labels
- `test_images.csv` - Test image list (no labels)
- `class_mapping.csv` - Class ID to scene name mapping
- `sample_submission.csv` - Example submission format

## Scene Categories

Examples: abbey, airport_terminal, bedroom, beach, forest, kitchen, mountain, 
office, restaurant, street, subway, etc.

## Submission Format

```csv
id,class_id
000001_class042.jpg,42
000002_class015.jpg,15
...
```

## Citation

J. Xiao, J. Hays, K. Ehinger, A. Oliva, and A. Torralba.
SUN Database: Large-scale Scene Recognition from Abbey to Zoo.
IEEE Conference on Computer Vision and Pattern Recognition, 2010.
"""
    
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(description='Prepare SUN397 for Kaggle (Final Eval)')
    parser.add_argument('--download_dir', type=str, default='./raw_data',
                        help='Directory to cache HuggingFace dataset')
    parser.add_argument('--output_dir', type=str, default='./kaggle_data_sun397',
                        help='Output directory for Kaggle dataset')
    parser.add_argument('--resolution', type=int, default=96,
                        help='Target image resolution (96, 224, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SUN397 Kaggle Competition Preparation (Final Evaluation)")
    print("="*60)
    print("\nInstalling requirements:")
    print("  pip install datasets")
    print()
    
    # Download SUN397 from HuggingFace
    dataset = download_sun397(args.download_dir)
    
    # Load dataset structure
    class_info, train_data = load_sun397_structure(dataset)
    
    # Create Kaggle dataset (resize to 96px for memory efficiency!)
    train_df, val_df, test_df = create_kaggle_dataset(
        train_data,
        args.output_dir,
        class_info,
        resolution=args.resolution,
        seed=args.seed
    )
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"Data saved to: {args.output_dir}")
    print("\nThis dataset is for evaluation.")


if __name__ == "__main__":
    main()

