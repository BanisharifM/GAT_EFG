#!/usr/bin/env python3
"""
Integrate CFG features into existing processed data
Adds CFG embeddings as additional columns
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

def load_cfg_features(cfg_json_path):
    """Load extracted CFG features"""
    with open(cfg_json_path, 'r') as f:
        cfg_data = json.load(f)
    return cfg_data

def select_features_for_modeling(cfg_data):
    """
    Select the most relevant features for modeling
    Returns: dict mapping benchmark -> feature vector
    """
    # Define which features to use (20-30 features)
    selected_features = [
        # Structural complexity
        'num_loops',
        'max_loop_depth',
        'num_branches',
        'cyclomatic_complexity',
        
        # Computational characteristics
        'num_arithmetic_ops',
        'num_memory_ops',
        'arithmetic_intensity',
        'memory_intensity',
        
        # Control flow
        'branch_density',
        'loop_intensity',
        
        # Memory patterns
        'num_array_accesses',
        'array_access_ratio',
        'num_pointer_ops',
        'pointer_complexity',
        
        # Function characteristics
        'num_function_calls',
        'call_overhead',
        'has_recursion',
        
        # Parallelism hints
        'has_parallel_pragma',
        
        # Code size proxy
        'source_lines',
    ]
    
    # Extract feature vectors
    benchmark_features = {}
    
    for benchmark, features in cfg_data.items():
        feature_vector = []
        for feat_name in selected_features:
            value = features.get(feat_name, 0)
            feature_vector.append(value)
        
        benchmark_features[benchmark] = feature_vector
    
    return benchmark_features, selected_features

def add_cfg_features_to_dataframe(df, cfg_features, feature_names):
    """
    Add CFG features as new columns to existing dataframe
    
    Args:
        df: existing dataframe with 'benchmark' column (label-encoded)
        cfg_features: dict mapping benchmark_name -> feature_vector
        feature_names: list of feature names
    
    Returns:
        Modified dataframe with cfg_* columns
    """
    # First, we need to map label-encoded benchmarks back to names
    # This requires the metadata from processing
    
    # For each row, look up the CFG features based on benchmark name
    for i, feat_name in enumerate(feature_names):
        col_name = f'cfg_{feat_name}'
        df[col_name] = 0.0  # Initialize
    
    # Map benchmark labels to names if metadata available
    # This is a placeholder - you'll need to adapt based on your encoding
    
    print(f"Added {len(feature_names)} CFG features as columns: cfg_*")
    return df

def integrate_with_processed_data(processed_csv, cfg_json, metadata_json, output_csv):
    """
    Main integration function
    
    Args:
        processed_csv: Path to processed data (e.g., data/splits/train_log.csv)
        cfg_json: Path to CFG features JSON
        metadata_json: Path to metadata with label encodings
        output_csv: Output path for augmented data
    """
    print("Loading data...")
    df = pd.read_csv(processed_csv)
    cfg_data = load_cfg_features(cfg_json)
    
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)
    
    # Get benchmark name mapping (label -> name)
    benchmark_mapping = metadata.get('benchmark_mapping', {})
    # Reverse mapping: label -> name
    label_to_name = {v: k for k, v in benchmark_mapping.items()}
    
    # Extract and select features
    benchmark_features, feature_names = select_features_for_modeling(cfg_data)
    
    print(f"Selected {len(feature_names)} CFG features")
    print(f"Features: {feature_names[:5]}... (showing first 5)")
    
    # Add CFG features to dataframe
    for i, feat_name in enumerate(feature_names):
        col_name = f'cfg_{feat_name}'
        
        # For each row, get the benchmark label, map to name, get CFG features
        def get_cfg_feature(row):
            benchmark_label = row['benchmark']
            benchmark_name = label_to_name.get(benchmark_label, None)
            
            if benchmark_name and benchmark_name in benchmark_features:
                return benchmark_features[benchmark_name][i]
            else:
                return 0.0  # Default if not found
        
        df[col_name] = df.apply(get_cfg_feature, axis=1)
    
    # Save augmented data
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved augmented data to: {output_csv}")
    print(f"  Original columns: {len(df.columns) - len(feature_names)}")
    print(f"  Added CFG columns: {len(feature_names)}")
    print(f"  Total columns: {len(df.columns)}")
    
    return df

def integrate_all_splits(cfg_json_path, data_dir='data'):
    """
    Integrate CFG features into train/val/test splits
    """
    data_dir = Path(data_dir)
    splits_dir = data_dir / 'splits'
    processed_dir = data_dir / 'processed'
    
    # Load metadata
    metadata_path = processed_dir / 'merged_metadata_processed.json'
    
    if not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        return
    
    # Process each split
    for split_name in ['train_log', 'val_log', 'test_log']:
        input_csv = splits_dir / f'{split_name}.csv'
        output_csv = splits_dir / f'{split_name}_with_cfg.csv'
        
        if input_csv.exists():
            print(f"\n{'='*60}")
            print(f"Processing {split_name}...")
            print('='*60)
            
            integrate_with_processed_data(
                input_csv, 
                cfg_json_path, 
                metadata_path, 
                output_csv
            )
        else:
            print(f"Warning: {input_csv} not found, skipping...")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate CFG features into datasets')
    parser.add_argument(
        '--cfg_json',
        type=str,
        default='cfg_analysis/features/cfg_features.json',
        help='Path to extracted CFG features JSON'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Root data directory'
    )
    
    args = parser.parse_args()
    
    # Integrate into all splits
    integrate_all_splits(args.cfg_json, args.data_dir)
    
    print("\n" + "="*60)
    print("INTEGRATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Update your dataset.py to load the *_with_cfg.csv files")
    print("2. Include cfg_* features in node or global features")
    print("3. Re-normalize data if needed")
    print("4. Retrain models with augmented features")


if __name__ == "__main__":
    main()