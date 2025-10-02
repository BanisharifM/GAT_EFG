#!/usr/bin/env python3
"""
Preprocessing script for RubikPi-GAT project.
Processes individual datasets and merges them.

Usage:
    python scripts/preprocess_data.py --config configs/preprocess_config.yaml
    python scripts/preprocess_data.py --config configs/preprocess_config.yaml --dataset rubikpi
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocess import DataPreprocessor, merge_datasets


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_single_dataset(config: dict, dataset_name: str):
    """Process a single dataset."""
    if dataset_name not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in config")
    
    dataset_config = config['datasets'][dataset_name]
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Process
    preprocessor.process_dataset(
        input_path=dataset_config['raw_data'],
        output_processed=dataset_config['processed_data'],
        output_normalized=dataset_config['normalized_data'],
        output_metadata=dataset_config['metadata'],
        fit_scaler=True
    )


def process_all_datasets(config: dict):
    """Process all datasets and merge them."""
    
    # Initialize preprocessor (will be reused for consistent encoding)
    preprocessor = DataPreprocessor(config)
    
    processed_paths = []
    normalized_paths = []
    
    # Process each dataset
    for dataset_name, dataset_config in config['datasets'].items():
        print(f"\n{'#'*60}")
        print(f"# Processing: {dataset_name.upper()}")
        print(f"{'#'*60}\n")
        
        # Check if raw file exists
        raw_path = Path(dataset_config['raw_data'])
        if not raw_path.exists():
            print(f"Warning: {raw_path} not found, skipping {dataset_name}")
            continue
        
        # Process dataset (only fit scaler on first dataset)
        fit_scaler = (len(processed_paths) == 0)
        
        df_proc, df_norm = preprocessor.process_dataset(
            input_path=dataset_config['raw_data'],
            output_processed=dataset_config['processed_data'],
            output_normalized=dataset_config['normalized_data'],
            output_metadata=dataset_config['metadata'],
            fit_scaler=fit_scaler
        )
        
        processed_paths.append(dataset_config['processed_data'])
        normalized_paths.append(dataset_config['normalized_data'])
    
    # Merge all datasets
    if len(processed_paths) > 1:
        print(f"\n{'#'*60}")
        print("# Merging All Datasets")
        print(f"{'#'*60}\n")
        
        merge_datasets(
            processed_paths=processed_paths,
            normalized_paths=normalized_paths,
            output_processed=config['merged']['processed_data'],
            output_normalized=config['merged']['normalized_data']
        )
    
    print(f"\n{'='*60}")
    print("✅ Preprocessing complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess RubikPi-GAT datasets"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/preprocess_config.yaml',
        help='Path to preprocessing config file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['rubikpi', 'tx2', 'nx', 'all'],
        default='all',
        help='Which dataset to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    
    # Process datasets
    if args.dataset == 'all':
        process_all_datasets(config)
    else:
        process_single_dataset(config, args.dataset)


if __name__ == '__main__':
    main()