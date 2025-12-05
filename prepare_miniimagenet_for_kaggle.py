"""
Mini-ImageNet Dataset Preparation for Kaggle Competition
========================================================

Creates Kaggle competition from Mini-ImageNet:
- 100 classes from ImageNet
- 60k images (600 per class)
- Resize from 84x84 to 96x96 or 224x224

Usage:
    python prepare_miniimagenet_for_kaggle.py \
        --output_dir ./kaggle_data_miniimagenet \
        --resolution 96
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import pickle
from PIL import Image
from tqdm import tqdm
import argparse


def download_mini_imagenet(download_dir):
    """
    Download Mini-ImageNet dataset
    
    Note: Mini-ImageNet requires downloading from multiple sources.
    The most common source is from:
    https://github.com/yaoyao-liu/mini-imagenet-tools
    
    Alternatively, you can use:
    https://www.kaggle.com/datasets/whitemoon/miniimagenet
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Mini-ImageNet Download Instructions")
    print("="*60)
    print()
    print("Option 1: Download from GitHub (recommended)")
    print("  https://github.com/yaoyao-liu/mini-imagenet-tools")
    print("  Files needed:")
    print("    - mini-imagenet-cache-train.pkl")
    print("    - mini-imagenet-cache-val.pkl")
    print("    - mini-imagenet-cache-test.pkl")
    print()
    print("Option 2: Download from Kaggle")
    print("  https://www.kaggle.com/datasets/whitemoon/miniimagenet")
    print("  Use kaggle API:")
    print("    kaggle datasets download -d whitemoon/miniimagenet")
    print()
    print("Option 3: Use torchvision (if available)")
    print()
    print("Please download the files and place them in:")
    print(f"  {download_dir}")
    print()
    
    # Check if files exist
    required_files = [
        'mini-imagenet-cache-train.pkl',
        'mini-imagenet-cache-val.pkl',
        'mini-imagenet-cache-test.pkl'
    ]
    
    all_exist = all((download_dir / f).exists() for f in required_files)
    
    if all_exist:
        print("✓ All Mini-ImageNet files found!")
        return download_dir
    else:
        print("⚠ Mini-ImageNet files not found.")
        print("Please download manually and place in:", download_dir)
        print()
        
        # Try to use learn2learn library if available
        try:
            import learn2learn as l2l
            print("Detected learn2learn library. Downloading via l2l...")
            return download_via_learn2learn(download_dir)
        except ImportError:
            print("\nTo auto-download, install: pip install learn2learn")
            print("Or download manually from the links above.")
            raise FileNotFoundError(f"Mini-ImageNet files not found in {download_dir}")


def download_via_learn2learn(download_dir):
    """Download using learn2learn library"""
    import learn2learn as l2l
    from torchvision import transforms
    
    print("\nDownloading Mini-ImageNet using learn2learn...")
    download_dir = Path(download_dir)
    
    # Download dataset
    dataset = l2l.vision.datasets.MiniImagenet(
        root=str(download_dir),
        mode='train',
        transform=None,
        download=True
    )
    
    print("Download complete!")
    return download_dir


def load_mini_imagenet_pkl(pkl_path):
    """Load Mini-ImageNet pickle file"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Structure: {'image_data': array, 'class_dict': dict}
    images = data['image_data']  # Shape: (N, 84, 84, 3)
    class_dict = data['class_dict']  # Dict: class_name -> [indices]
    
    return images, class_dict


def create_kaggle_dataset(
    download_dir,
    output_dir,
    resolution=96,
    seed=42
):
    """
    Create Kaggle competition dataset from Mini-ImageNet
    
    Args:
        download_dir: Path to Mini-ImageNet pkl files
        output_dir: Output directory
        resolution: Target resolution (96, 224, etc.)
        seed: Random seed
    """
    download_dir = Path(download_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(seed)
    
    print("\n" + "="*60)
    print("Loading Mini-ImageNet dataset...")
    print("="*60)
    
    # Load train split (required)
    train_images, train_class_dict = load_mini_imagenet_pkl(
        download_dir / 'mini-imagenet-cache-train.pkl'
    )
    
    # Load val split (optional)
    val_path = download_dir / 'mini-imagenet-cache-val.pkl'
    if val_path.exists():
        val_images, val_class_dict = load_mini_imagenet_pkl(val_path)
    else:
        print("  Val file not found, skipping...")
        val_images = np.array([]).reshape(0, 84, 84, 3)
        val_class_dict = {}
    
    # Load test split (optional)
    test_path = download_dir / 'mini-imagenet-cache-test.pkl'
    if test_path.exists():
        test_images, test_class_dict = load_mini_imagenet_pkl(test_path)
    else:
        print("  Test file not found, skipping...")
        test_images = np.array([]).reshape(0, 84, 84, 3)
        test_class_dict = {}
    
    print(f"\nOriginal splits:")
    print(f"  Train: {len(train_class_dict)} classes, {len(train_images)} images")
    if len(val_class_dict) > 0:
        print(f"  Val: {len(val_class_dict)} classes, {len(val_images)} images")
    if len(test_class_dict) > 0:
        print(f"  Test: {len(test_class_dict)} classes, {len(test_images)} images")
    
    # Combine all data
    all_images_list = [train_images]
    if len(val_images) > 0:
        all_images_list.append(val_images)
    if len(test_images) > 0:
        all_images_list.append(test_images)
    all_images = np.concatenate(all_images_list, axis=0)
    
    all_classes = {}
    offset = 0
    
    # Add train classes
    for class_name, indices in train_class_dict.items():
        all_classes[class_name] = [i + offset for i in indices]
    offset += len(train_images)
    
    # Add val classes (if exist)
    if len(val_class_dict) > 0:
        for class_name, indices in val_class_dict.items():
            if class_name in all_classes:
                all_classes[class_name].extend([i + offset for i in indices])
            else:
                all_classes[class_name] = [i + offset for i in indices]
        offset += len(val_images)
    
    # Add test classes (if exist)
    if len(test_class_dict) > 0:
        for class_name, indices in test_class_dict.items():
            if class_name in all_classes:
                all_classes[class_name].extend([i + offset for i in indices])
            else:
                all_classes[class_name] = [i + offset for i in indices]
    
    print(f"\nCombined dataset:")
    print(f"  Total classes: {len(all_classes)}")
    print(f"  Total images: {len(all_images)}")
    print(f"  Image shape: {all_images.shape}")
    
    # Create class mapping
    class_names = sorted(all_classes.keys())
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"\nSample classes:")
    for name in class_names[:10]:
        print(f"  {name}: {len(all_classes[name])} images")
    print("  ...")
    
    # Collect all data with labels
    all_data = []
    for class_name in class_names:
        class_id = class_mapping[class_name]
        for img_idx in all_classes[class_name]:
            all_data.append({
                'image_idx': img_idx,
                'class_id': class_id,
                'class_name': class_name
            })
    
    # Shuffle and split
    np.random.shuffle(all_data)
    
    n_total = len(all_data)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]
    
    print(f"\nNew splits for Kaggle:")
    print(f"  Train: {len(train_data)} images")
    print(f"  Val: {len(val_data)} images")
    print(f"  Test: {len(test_data)} images")
    
    # Save images
    def save_split(split_data, split_name):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        print(f"\nSaving {split_name} images (resolution={resolution}x{resolution})...")
        for idx, item in enumerate(tqdm(split_data, desc=f"{split_name}")):
            # Get image from array
            img_array = all_images[item['image_idx']]  # Shape: (84, 84, 3)
            
            # Convert to PIL and resize
            img = Image.fromarray(img_array.astype('uint8'))
            img = img.resize((resolution, resolution), Image.BILINEAR)
            
            # Save with new filename
            filename = f"{idx:05d}_class{item['class_id']:03d}.jpg"
            img.save(split_dir / filename, quality=95)
            
            metadata.append({
                'filename': filename,
                'class_id': item['class_id'],
                'class_name': item['class_name']
            })
        
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
    
    # Create README
    create_readme(output_dir, len(class_names), resolution)
    
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
    print(f"Sample submission created")


def create_readme(output_dir, num_classes, resolution):
    """Create README for dataset"""
    readme = f"""# Mini-ImageNet Kaggle Competition Dataset

## Dataset Info

- **Source**: Mini-ImageNet (subset of ImageNet for few-shot learning)
- **Classes**: {num_classes} object categories from ImageNet
- **Resolution**: {resolution}x{resolution} (upscaled from 84x84)
- **Total Images**: ~60k (600 per class)

## Files

- `train/` - Training images with labels
- `val/` - Validation images with labels
- `test/` - Test images (no labels for students)
- `train_labels.csv` - Training labels
- `val_labels.csv` - Validation labels
- `test_images.csv` - Test image list (no labels)
- `sample_submission.csv` - Example submission format

## Submission Format

```csv
id,class_id
00001_class042.jpg,42
00002_class015.jpg,15
...
```

## Citation

Mini-ImageNet dataset for few-shot learning
"""
    
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(description='Prepare Mini-ImageNet for Kaggle')
    parser.add_argument('--download_dir', type=str, default='./raw_data',
                        help='Directory containing Mini-ImageNet pkl files')
    parser.add_argument('--output_dir', type=str, default='./kaggle_data_miniimagenet',
                        help='Output directory for Kaggle dataset')
    parser.add_argument('--resolution', type=int, default=96,
                        help='Target image resolution (96, 224, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Mini-ImageNet Kaggle Competition Preparation")
    print("="*60)
    
    # Download/check Mini-ImageNet
    download_dir = download_mini_imagenet(args.download_dir)
    
    # Create Kaggle dataset
    train_df, val_df, test_df = create_kaggle_dataset(
        download_dir,
        args.output_dir,
        resolution=args.resolution,
        seed=args.seed
    )
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"Data saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

