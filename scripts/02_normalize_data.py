#!/usr/bin/env python3
"""
Step 2: Data Normalization

Normalizes processed CSV files using StandardScaler.
Fits scaler on specified data (single file or merged).

Usage:
    # Normalize single file
    python scripts/02_normalize_data.py --input data/processed/rubikpi_processed.csv --output data/processed/rubikpi_normalized.csv
    
    # Normalize using merged data for scaler fitting
    python scripts/02_normalize_data.py --config configs/preprocess_config.yaml --use-merged
    
    # Normalize all individual files using merged scaler
    python scripts/02_normalize_data.py --config configs/preprocess_config.yaml --fit-on-merged
"""

import argparse
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataNormalizer:
    """Normalize processed data."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config
        self.scaler = StandardScaler()
        self.scaler_params = {}
        self.columns_to_normalize = []
    
    def fit_scaler(self, df: pd.DataFrame, columns: List[str]) -> None:
        """Fit scaler on data."""
        self.columns_to_normalize = [col for col in columns if col in df.columns]
        
        print(f"Fitting scaler on {len(df)} samples, {len(self.columns_to_normalize)} columns")
        
        self.scaler.fit(df[self.columns_to_normalize])
        
        self.scaler_params = {
            'mean': self.scaler.mean_.tolist(),
            'std': self.scaler.scale_.tolist(),
            'columns': self.columns_to_normalize,
            'n_samples': len(df)
        }
        
        print(f"  ✅ Scaler fitted")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if not self.columns_to_normalize:
            raise ValueError("Scaler not fitted! Call fit_scaler() first.")
        
        df_norm = df.copy()
        df_norm[self.columns_to_normalize] = self.scaler.transform(
            df[self.columns_to_normalize]
        )
        
        return df_norm
    
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit scaler and transform in one step."""
        self.fit_scaler(df, columns)
        return self.transform(df)
    
    def save_metadata(self, filepath: str) -> None:
        """Save scaler metadata."""
        metadata = {
            'scaler_params': self.to_json_serializable(self.scaler_params),
            'normalization_method': 'standard',
            'columns_normalized': self.columns_to_normalize
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Metadata: {filepath}")
    
    @staticmethod
    def to_json_serializable(obj):
        """Convert numpy types to JSON serializable."""
        if isinstance(obj, dict):
            return {k: DataNormalizer.to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DataNormalizer.to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def normalize_single_file(input_path: str, output_path: str, 
                          columns_to_normalize: List[str],
                          metadata_path: Optional[str] = None) -> None:
    """Normalize a single processed CSV file."""
    print(f"\n{'='*60}")
    print(f"Normalizing: {Path(input_path).name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df)} rows")
    
    normalizer = DataNormalizer()
    df_norm = normalizer.fit_transform(df, columns_to_normalize)
    
    df_norm.to_csv(output_path, index=False, float_format='%.6f')
    print(f"  ✅ Saved: {output_path}")
    
    if metadata_path:
        normalizer.save_metadata(metadata_path)


def normalize_with_config(config: dict, fit_on_merged: bool = False) -> None:
    """Normalize datasets using config file."""
    
    # Get columns to normalize
    cols_to_norm = config['normalization']['numerical_features'] + \
                   config['normalization']['target']
    
    normalizer = DataNormalizer(config)
    
    if fit_on_merged:
        # Strategy: Fit on merged, apply to all
        print(f"\n{'#'*60}")
        print("# Strategy: Fit scaler on MERGED data")
        print(f"{'#'*60}\n")
        
        # Load merged processed data
        merged_path = config['merged']['processed_data']
        print(f"Loading merged data: {merged_path}")
        df_merged = pd.read_csv(merged_path)
        
        # Fit scaler on merged
        normalizer.fit_scaler(df_merged, cols_to_norm)
        
        # Normalize merged
        df_merged_norm = normalizer.transform(df_merged)
        merged_norm_path = config['merged']['normalized_data']
        df_merged_norm.to_csv(merged_norm_path, index=False, float_format='%.6f')
        print(f"  ✅ Saved merged: {merged_norm_path}")
        
        # Normalize individual datasets with same scaler
        print(f"\nNormalizing individual datasets with merged scaler:")
        for dataset_name, dataset_config in config['datasets'].items():
            proc_path = Path(dataset_config['processed_data'])
            
            if not proc_path.exists():
                continue
            
            df = pd.read_csv(proc_path)
            df_norm = normalizer.transform(df)
            
            norm_path = dataset_config['normalized_data']
            df_norm.to_csv(norm_path, index=False, float_format='%.6f')
            print(f"  ✅ {dataset_name}: {norm_path}")
        
        # Save metadata
        metadata_path = config['merged']['metadata']
        normalizer.save_metadata(metadata_path)
        
    else:
        # Strategy: Normalize each dataset independently
        print(f"\n{'#'*60}")
        print("# Strategy: Independent normalization per dataset")
        print(f"{'#'*60}\n")
        
        for dataset_name, dataset_config in config['datasets'].items():
            proc_path = Path(dataset_config['processed_data'])
            
            if not proc_path.exists():
                print(f"⚠️  Skipping {dataset_name}: {proc_path} not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"Normalizing: {dataset_name}")
            print(f"{'='*60}")
            
            df = pd.read_csv(proc_path)
            
            norm = DataNormalizer()
            df_norm = norm.fit_transform(df, cols_to_norm)
            
            norm_path = dataset_config['normalized_data']
            df_norm.to_csv(norm_path, index=False, float_format='%.6f')
            print(f"  ✅ Saved: {norm_path}")
            
            meta_path = dataset_config['metadata']
            norm.save_metadata(meta_path)
    
    print(f"\n{'='*60}")
    print("✅ Step 2 Complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Normalize processed data (Step 2)")
    parser.add_argument('--config', type=str, 
                       help='Config file path')
    parser.add_argument('--input', type=str,
                       help='Input processed CSV file')
    parser.add_argument('--output', type=str,
                       help='Output normalized CSV file')
    parser.add_argument('--fit-on-merged', action='store_true',
                       help='Fit scaler on merged data, apply to all')
    
    args = parser.parse_args()
    
    if args.config:
        # Config-based normalization
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        normalize_with_config(config, fit_on_merged=args.fit_on_merged)
        
    elif args.input and args.output:
        # Single file normalization
        # Load config to get columns to normalize
        config_path = 'configs/preprocess_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        cols_to_norm = config['normalization']['numerical_features'] + \
                       config['normalization']['target']
        
        metadata_path = args.output.replace('.csv', '_metadata.json')
        
        normalize_single_file(args.input, args.output, cols_to_norm, metadata_path)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()