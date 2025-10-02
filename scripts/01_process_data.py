#!/usr/bin/env python3
"""
Step 1: Data Processing (NO normalization)

Processes raw CSV files:
- Parses array columns (frequency_combination, frequencies)
- Encodes categorical variables (run_mode, benchmark, platform)
- Parses special columns (core_combination → binary mask)
- Aggregates thermal zones
- Cleans data (duplicates, missing values, constraints)

Outputs:
- Individual processed CSVs
- Merged processed CSV (if multiple files)
- Metadata JSON (encodings, column info)

Usage:
    python scripts/01_process_data.py --config configs/preprocess_config.yaml
"""

import argparse
import yaml
import json
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import Dict, List, Any
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Process raw data without normalization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.label_encoders = {}
    
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """Load raw CSV data."""
        print(f"Loading: {filepath}")
        df = pd.read_csv(filepath)
        print(f"  → {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def parse_array_column(self, df: pd.DataFrame, column: str, 
                          config: Dict[str, Any]) -> pd.DataFrame:
        """Parse array string columns into separate columns."""
        prefix = config['output_prefix']
        length = config['length']
        dtype = config['dtype']
        
        def safe_parse(x):
            try:
                if isinstance(x, str):
                    return ast.literal_eval(x)
                return x
            except:
                return [0] * length
        
        parsed = df[column].apply(safe_parse)
        
        for i in range(length):
            df[f"{prefix}_{i}"] = parsed.apply(
                lambda x: x[i] if isinstance(x, list) and len(x) > i else 0
            ).astype(dtype)
        
        return df
    
    def parse_core_combination(self, df: pd.DataFrame, 
                               config: Dict[str, Any]) -> pd.DataFrame:
        """Parse core_combination into binary mask."""
        prefix = config['output_prefix']
        length = config['length']
        dtype = config['dtype']
        
        def to_binary_mask(core_str):
            mask = [0] * length
            if pd.isna(core_str):
                return mask
            cores = [int(c) for c in str(core_str) if c.isdigit()]
            for core_id in cores:
                if 1 <= core_id <= length:
                    mask[core_id - 1] = 1
            return mask
        
        masks = df['core_combination'].apply(to_binary_mask)
        for i in range(length):
            df[f"{prefix}_{i}"] = masks.apply(lambda x: x[i]).astype(dtype)
        
        return df
    
    def parse_cores_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse cores_str to count number of cores."""
        def count_cores(core_str):
            if pd.isna(core_str):
                return 0
            core_str = str(core_str)
            if ',' in core_str:
                return len(core_str.split(','))
            elif core_str.strip():
                return 1
            else:
                return 0
        
        df['cores_str'] = df['cores_str'].apply(count_cores).astype('int8')
        return df
    
    def aggregate_thermal_zones(self, df: pd.DataFrame, 
                                config: Dict[str, Any]) -> pd.DataFrame:
        """Aggregate thermal zones into single mean value."""
        prefix = config['input_prefix']
        output_col = config['output_column']
        dtype = config['dtype']
        
        thermal_cols = [col for col in df.columns if col.startswith(prefix)]
        
        if not thermal_cols:
            df[output_col] = 0.0
            return df
        
        def thermal_mean(row):
            values = row[thermal_cols].values
            non_zero = values[values > 0]
            return non_zero.mean() if len(non_zero) > 0 else 0.0
        
        df[output_col] = df.apply(thermal_mean, axis=1).astype(dtype)
        return df
    
    def encode_categorical(self, df: pd.DataFrame, 
                          config: Dict[str, Any]) -> pd.DataFrame:
        """Encode categorical variables."""
        for feature, feat_config in config.items():
            if feature not in df.columns:
                continue
            
            # Clean whitespace
            df[feature] = df[feature].astype(str).str.strip()
            dtype = feat_config['dtype']
            
            if 'mapping' in feat_config:
                mapping = feat_config['mapping']
                df[feature] = df[feature].map(mapping)
                
                # Warn about unmapped values
                unmapped = df[feature].isna().sum()
                if unmapped > 0:
                    unique_vals = df[df[feature].isna()][feature].unique()
                    print(f"  ⚠️  {unmapped} unmapped '{feature}': {list(unique_vals)[:3]}")
                
                df[feature] = df[feature].fillna(-1).astype(dtype)
                self.label_encoders[feature] = mapping
                
            elif feat_config.get('auto_detect', False):
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
                df[feature] = df[feature].astype(dtype)
                self.label_encoders[feature] = dict(
                    zip(le.classes_, le.transform(le.classes_))
                )
        
        return df
    
    def cast_numerical_types(self, df: pd.DataFrame, 
                            config: Dict[str, Any]) -> pd.DataFrame:
        """Cast numerical columns to specified dtypes."""
        for feature, feat_config in config.items():
            if feature in df.columns:
                df[feature] = df[feature].astype(feat_config['dtype'])
        return df
    
    def clean_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Clean data."""
        initial = len(df)
        
        if config.get('remove_duplicates', True):
            df = df.drop_duplicates()
        
        if config.get('drop_na', True):
            df = df.dropna()
        
        if 'constraints' in config:
            for col, limits in config['constraints'].items():
                if col in df.columns:
                    if 'min' in limits:
                        df = df[df[col] >= limits['min']]
                    if 'max' in limits:
                        df = df[df[col] <= limits['max']]
        
        removed = initial - len(df)
        if removed > 0:
            print(f"  → Removed {removed} rows (duplicates/NaN/constraints)")
        
        return df
    
    def get_final_columns(self) -> List[str]:
        """Get final column order."""
        cols = []
        cols.extend(self.config['features']['categorical'].keys())
        cols.extend(self.config['features']['numerical'].keys())
        cols.append(self.config['features']['thermal']['output_column'])
        
        for feat_config in self.config['features']['arrays'].values():
            prefix = feat_config['output_prefix']
            length = feat_config['length']
            cols.extend([f"{prefix}_{i}" for i in range(length)])
        
        special_config = self.config['features']['special']['core_combination']
        prefix = special_config['output_prefix']
        length = special_config['length']
        cols.extend([f"{prefix}_{i}" for i in range(length)])
        
        for target in self.config['targets']:
            cols.append(target['name'])
        
        return cols
    
    def process_single_file(self, input_path: str, output_path: str) -> pd.DataFrame:
        """Process a single CSV file."""
        print(f"\n{'='*60}")
        print(f"Processing: {Path(input_path).name}")
        print(f"{'='*60}")
        
        df = self.load_raw_data(input_path)
        
        # Parse arrays
        for feature, feat_config in self.config['features']['arrays'].items():
            if feature in df.columns:
                df = self.parse_array_column(df, feature, feat_config)
        
        # Parse special features
        if 'core_combination' in df.columns:
            df = self.parse_core_combination(
                df, self.config['features']['special']['core_combination']
            )
        
        if 'cores_str' in df.columns:
            df = self.parse_cores_str(df)
        
        # Aggregate thermal
        df = self.aggregate_thermal_zones(df, self.config['features']['thermal'])
        
        # Encode categorical
        df = self.encode_categorical(df, self.config['features']['categorical'])
        
        # Cast types
        df = self.cast_numerical_types(df, self.config['features']['numerical'])
        
        # Clean
        df = self.clean_data(df, self.config['cleaning'])
        
        # Select final columns
        final_cols = self.get_final_columns()
        df_final = df[final_cols].copy()
        
        # Save
        df_final.to_csv(output_path, index=False, float_format='%.6f')
        print(f"  ✅ Saved: {output_path} ({len(df_final)} rows)")
        
        return df_final


def to_json_serializable(obj):
    """Convert numpy types to JSON serializable."""
    if isinstance(obj, dict):
        return {key: to_json_serializable(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="Process raw data (Step 1)")
    parser.add_argument('--config', type=str, 
                       default='configs/preprocess_config.yaml',
                       help='Config file path')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize processor
    processor = DataProcessor(config)
    
    # Process each dataset
    processed_dfs = []
    processed_paths = []
    
    print(f"\n{'#'*60}")
    print("# Step 1: Processing Raw Data")
    print(f"{'#'*60}\n")
    
    for dataset_name, dataset_config in config['datasets'].items():
        raw_path = Path(dataset_config['raw_data'])
        
        if not raw_path.exists():
            print(f"⚠️  Skipping {dataset_name}: {raw_path} not found")
            continue
        
        df = processor.process_single_file(
            input_path=str(raw_path),
            output_path=dataset_config['processed_data']
        )
        
        processed_dfs.append(df)
        processed_paths.append(dataset_config['processed_data'])
    
    # Merge if multiple datasets
    if len(processed_dfs) > 1:
        print(f"\n{'='*60}")
        print("Merging processed datasets")
        print(f"{'='*60}")
        
        df_merged = pd.concat(processed_dfs, ignore_index=True)
        merged_path = config['merged']['processed_data']
        df_merged.to_csv(merged_path, index=False, float_format='%.6f')
        
        print(f"  ✅ Merged: {merged_path} ({len(df_merged)} rows)")
    
    # Save metadata
    metadata = {
        'label_encoders': to_json_serializable(processor.label_encoders),
        'final_columns': processor.get_final_columns(),
        'num_datasets': len(processed_dfs),
        'total_samples': sum(len(df) for df in processed_dfs),
        'gat_structure': config['gat_structure']
    }
    
    metadata_path = config['merged']['metadata'].replace('.json', '_processed.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ Step 1 Complete!")
    print(f"{'='*60}")
    print(f"Processed files: {len(processed_paths)}")
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Metadata: {metadata_path}\n")


if __name__ == '__main__':
    main()