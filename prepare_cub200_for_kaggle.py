"""
CUB-200-2011 Dataset Preparation for Kaggle Competition
========================================================
[INTERNAL - FOR TAs ONLY]

This script prepares the CUB-200-2011 dataset for the Kaggle competition.
Run this script to:
1. Download CUB-200-2011 dataset
2. Create train/val/test splits
3. Generate solution.csv for Kaggle
4. Prepare data for HuggingFace upload

Usage:
    python prepare_cub200_for_kaggle.py --download_dir ./raw_data --output_dir ./kaggle_data
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import tarfile
from sklearn.model_selection import train_test_split
import argparse


def download_cub200(download_dir):
    """Download CUB-200-2011 dataset"""
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    tar_path = download_dir / "CUB_200_2011.tgz"
    
    if not tar_path.exists():
        print(f"Downloading CUB-200-2011 from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete!")
    
    # Extract
    extract_dir = download_dir / "CUB_200_2011"
    if not extract_dir.exists():
        print("Extracting dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(download_dir)
        print("Extraction complete!")
    
    return extract_dir


def create_splits(cub_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Create train/val/test splits from CUB-200-2011
    
    Args:
        cub_dir: Path to CUB_200_2011 directory
        output_dir: Output directory for splits
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set (will be split into public/private)
        seed: Random seed
    """
    cub_dir = Path(cub_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read metadata
    images_df = pd.read_csv(cub_dir / 'images.txt', sep=' ', names=['image_id', 'filepath'])
    labels_df = pd.read_csv(cub_dir / 'image_class_labels.txt', sep=' ', names=['image_id', 'class_id'])
    train_test_split_df = pd.read_csv(cub_dir / 'train_test_split.txt', sep=' ', names=['image_id', 'is_training_image'])
    classes_df = pd.read_csv(cub_dir / 'classes.txt', sep=' ', names=['class_id', 'class_name'])
    
    # Merge data
    data = images_df.merge(labels_df, on='image_id').merge(train_test_split_df, on='image_id')
    data = data.merge(classes_df, on='class_id')
    
    # Convert class_id to 0-indexed
    data['class_id'] = data['class_id'] - 1
    
    print(f"Total images: {len(data)}")
    print(f"Total classes: {data['class_id'].nunique()}")
    
    # Split data
    np.random.seed(seed)
    
    # Group by class to ensure balanced splits
    train_data = []
    val_data = []
    test_data = []
    
    for class_id in sorted(data['class_id'].unique()):
        class_data = data[data['class_id'] == class_id].copy()
        n_samples = len(class_data)
        
        # Shuffle
        class_data = class_data.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Split
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_data.append(class_data.iloc[:n_train])
        val_data.append(class_data.iloc[n_train:n_train+n_val])
        test_data.append(class_data.iloc[n_train+n_val:])
    
    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} images")
    print(f"Val: {len(val_df)} images")
    print(f"Test: {len(test_df)} images")
    
    # Copy images to output directory
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {split_name} images...")
        for idx, row in split_df.iterrows():
            src = cub_dir / 'images' / row['filepath']
            # Create flat structure with unique filenames
            dst_filename = f"{row['image_id']:05d}_{Path(row['filepath']).name}"
            dst = split_dir / dst_filename
            shutil.copy2(src, dst)
            split_df.at[idx, 'filename'] = dst_filename
    
    # Save metadata CSVs
    train_df[['filename', 'class_id', 'class_name']].to_csv(output_dir / 'train_labels.csv', index=False)
    val_df[['filename', 'class_id', 'class_name']].to_csv(output_dir / 'val_labels.csv', index=False)
    
    # For test: save full labels internally, but students won't get this
    test_df[['filename', 'class_id', 'class_name']].to_csv(output_dir / 'test_labels_INTERNAL.csv', index=False)
    
    # Create test metadata without labels (for students)
    test_df[['filename']].to_csv(output_dir / 'test_images.csv', index=False)
    
    return train_df, val_df, test_df



def create_sample_submission(test_df, output_dir):
    """Create sample_submission.csv for students"""
    output_dir = Path(output_dir)
    
    # Create random predictions as example
    sample_submission = pd.DataFrame({
        'id': test_df['filename'],
        'class_id': 0  # Dummy prediction (all class 0)
    })
    
    sample_path = output_dir / 'sample_submission.csv'
    sample_submission.to_csv(sample_path, index=False)
    
    print(f"\nSample submission created: {sample_path}")
    print(f"Format preview:")
    print(sample_submission.head(10))
    
    return sample_submission


def create_readme(output_dir):
    """Create README for the dataset"""
    output_dir = Path(output_dir)
    
    readme_content = """# CUB-200-2011 Dataset for SSL Final Project

## Dataset Structure

```
kaggle_data/
├── train/                  # Training images (WITH labels)
│   ├── 00001_image.jpg
│   └── ...
├── val/                    # Validation images (WITH labels)
│   ├── 00234_image.jpg
│   └── ...
├── test/                   # Test images (NO labels for students)
│   ├── 00456_image.jpg
│   └── ...
├── train_labels.csv        # Labels for training set
├── val_labels.csv          # Labels for validation set
├── test_images.csv         # List of test images (no labels)
└── sample_submission.csv   # Example submission format
```

## Files Description

### For Students:

1. **train/** and **train_labels.csv**: 
   - Training images and their labels
   - Use to train your linear classifier on frozen SSL features

2. **val/** and **val_labels.csv**: 
   - Validation images and labels
   - Use for hyperparameter tuning of linear classifier

3. **test/** and **test_images.csv**: 
   - Test images WITHOUT labels
   - Extract features and create predictions
   - Submit predictions to Kaggle

4. **sample_submission.csv**: 
   - Example format for Kaggle submission
   - Columns: `id` (filename), `class_id` (predicted class)

### For TAs Only (DO NOT SHARE):

- **test_labels_INTERNAL.csv**: Ground truth for test set
- **solution.csv**: Kaggle solution file with Public/Private splits

## Dataset Statistics

- **Classes**: 200 bird species
- **Total Images**: ~11,788
- **Train Split**: ~70% (~8,200 images)
- **Val Split**: ~15% (~1,750 images)
- **Test Split**: ~15% (~1,750 images)
- **Image Resolution**: Variable (resize to 224x224 recommended)

## Class Distribution

Each class (bird species) has approximately:
- 40-50 training images
- 8-10 validation images
- 8-10 test images

## Submission Format

Your `submission.csv` should have two columns:
```
id,class_id
00456_Black_footed_Albatross_0001_796111.jpg,0
00789_Laysan_Albatross_0002_545.jpg,15
...
```

Where:
- `id`: filename from test set
- `class_id`: predicted class (0-199)

## Evaluation Metric

**Accuracy Score**: `sklearn.metrics.accuracy_score`
- Fraction of correctly classified test images
- Higher is better

## Citation

```
@techreport{WelinderEtal2010,
    Author = {P. Welinder and S. Branson and T. Mita and C. Wah and F. Schroff and S. Belongie and P. Perona},
    Institution = {California Institute of Technology},
    Number = {CNS-TR-2010-001},
    Title = {{Caltech-UCSD Birds 200}},
    Year = {2010}
}
```
"""
    
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"\nREADME created: {output_dir / 'README.md'}")


def main():
    parser = argparse.ArgumentParser(description='Prepare CUB-200-2011 for Kaggle Competition')
    parser.add_argument('--download_dir', type=str, default='./raw_data',
                        help='Directory to download raw CUB-200 dataset')
    parser.add_argument('--output_dir', type=str, default='./kaggle_data',
                        help='Output directory for processed data')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip downloading if data already exists')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CUB-200-2011 Kaggle Competition Data Preparation")
    print("="*60)
    
    # Step 1: Download dataset
    if not args.skip_download:
        cub_dir = download_cub200(args.download_dir)
    else:
        cub_dir = Path(args.download_dir) / "CUB_200_2011"
        if not cub_dir.exists():
            raise FileNotFoundError(f"CUB directory not found: {cub_dir}")
    
    # Step 2: Create splits
    train_df, val_df, test_df = create_splits(cub_dir, args.output_dir, seed=args.seed)
    
    # Step 3: Create sample submission (solution.csv removed - for TAs only!)
    sample_submission = create_sample_submission(test_df, args.output_dir)
    
    # Step 4: Create README
    create_readme(args.output_dir)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"Review the data in: {args.output_dir}")



if __name__ == "__main__":
    main()

