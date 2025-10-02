#!/usr/bin/env python3
"""
Integrate CFG features into existing training/validation/test splits
"""

import pandas as pd
import json
from pathlib import Path
import sys

def load_cfg_features(cfg_csv_path):
    """Load CFG features from CSV"""
    df_cfg = pd.read_csv(cfg_csv_path)
    
    # Select numeric features for modeling (exclude metadata)
    feature_cols = [col for col in df_cfg.columns 
                   if col not in ['benchmark', 'source_file', 'source_lines']]
    
    print(f"Loaded {len(df_cfg)} benchmarks with {len(feature_cols)} CFG features")
    print(f"Features: {feature_cols[:5]}... (showing first 5)")
    
    return df_cfg, feature_cols

def get_benchmark_name_mapping(metadata_path):
    """Get mapping from encoded label to benchmark name"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # FIXED: Get benchmark encoding from correct key
    benchmark_encoding = metadata.get('label_encoders', {}).get('benchmark', {})
    
    # Reverse it: label -> name
    label_to_name = {int(v): k for k, v in benchmark_encoding.items()}
    
    print(f"\nFound {len(label_to_name)} benchmark mappings")
    print(f"Sample mappings: {dict(list(label_to_name.items())[:3])}")
    return label_to_name

def integrate_cfg_into_split(split_csv, cfg_df, feature_cols, label_to_name, output_csv):
    """Add CFG features to a train/val/test split"""
    
    # Load split data
    df_split = pd.read_csv(split_csv)
    print(f"\nProcessing {split_csv.name}:")
    print(f"  Original shape: {df_split.shape}")
    
    # Create a mapping from benchmark name to CFG features
    cfg_dict = {}
    for _, row in cfg_df.iterrows():
        benchmark_name = row['benchmark']
        cfg_dict[benchmark_name] = row[feature_cols].values
    
    # Add CFG features for each row
    missing_count = 0
    for i, feat_name in enumerate(feature_cols):
        col_name = f'cfg_{feat_name}'
        
        # For each row, get benchmark label -> name -> CFG features
        def get_cfg_value(benchmark_label):
            nonlocal missing_count
            benchmark_name = label_to_name.get(int(benchmark_label), None)
            if benchmark_name and benchmark_name in cfg_dict:
                return cfg_dict[benchmark_name][i]
            else:
                if i == 0:  # Only count once per row
                    missing_count += 1
                return 0.0  # Default if not found
        
        df_split[col_name] = df_split['benchmark'].apply(get_cfg_value)
    
    # Save augmented data
    df_split.to_csv(output_csv, index=False)
    
    print(f"  New shape: {df_split.shape}")
    print(f"  Added {len(feature_cols)} CFG columns")
    print(f"  Missing mappings: {missing_count} rows")
    print(f"  Saved to: {output_csv.name}")
    
    return df_split

def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    cfg_csv = project_root / 'cfg_analysis/features/cfg_features.csv'
    metadata_json = project_root / 'data/processed/merged_metadata_processed.json'
    splits_dir = project_root / 'data/splits'
    
    # Verify files exist
    if not cfg_csv.exists():
        print(f"Error: CFG features not found at {cfg_csv}")
        sys.exit(1)
    
    if not metadata_json.exists():
        print(f"Error: Metadata not found at {metadata_json}")
        sys.exit(1)
    
    # Load CFG features
    cfg_df, feature_cols = load_cfg_features(cfg_csv)
    
    # Load benchmark name mapping
    label_to_name = get_benchmark_name_mapping(metadata_json)
    
    # Process each split
    splits = ['train_log', 'val_log', 'test_log']
    
    for split_name in splits:
        input_csv = splits_dir / f'{split_name}.csv'
        output_csv = splits_dir / f'{split_name}_with_cfg.csv'
        
        if input_csv.exists():
            integrate_cfg_into_split(input_csv, cfg_df, feature_cols, label_to_name, output_csv)
        else:
            print(f"Warning: {input_csv} not found, skipping...")
    
    print("\n" + "="*60)
    print("INTEGRATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Update src/data/dataset.py to load *_with_cfg.csv files")
    print("2. Add cfg_* columns to global_feature_cols")
    print("3. Retrain your model!")

if __name__ == "__main__":
    main()
