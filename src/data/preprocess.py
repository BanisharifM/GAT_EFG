"""
Data preprocessing functions for RubikPi-GAT project.
Handles feature engineering, encoding, and normalization.
"""

import pandas as pd
import numpy as np
import json
import ast
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Dictionary containing preprocessing configuration
        """
        self.config = config
        self.label_encoders = {}
        self.scaler = None
        self.scaler_params = {}
        
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """Load raw CSV data."""
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def parse_array_column(self, df: pd.DataFrame, column: str, 
                          config: Dict[str, Any]) -> pd.DataFrame:
        """
        Parse array-like string columns into separate columns.
        
        Example: "[0,0,0,0,0,0,0,0]" -> freq_lvl_0, freq_lvl_1, ..., freq_lvl_7
        """
        print(f"Parsing array column: {column}")
        
        prefix = config['output_prefix']
        length = config['length']
        dtype = config['dtype']
        
        # Parse string arrays
        def safe_parse(x):
            try:
                if isinstance(x, str):
                    return ast.literal_eval(x)
                return x
            except:
                return [0] * length
        
        parsed = df[column].apply(safe_parse)
        
        # Create separate columns
        for i in range(length):
            df[f"{prefix}_{i}"] = parsed.apply(
                lambda x: x[i] if isinstance(x, list) and len(x) > i else 0
            ).astype(dtype)
        
        return df
    
    def parse_core_combination(self, df: pd.DataFrame, 
                               config: Dict[str, Any]) -> pd.DataFrame:
        """
        Parse core_combination into binary mask.
        
        Example: "123" -> [1,1,1,0,0,0,0,0]
                 "1357" -> [1,0,1,0,1,0,1,0]
        """
        print("Parsing core_combination to binary mask")
        
        prefix = config['output_prefix']
        length = config['length']
        dtype = config['dtype']
        
        def to_binary_mask(core_str):
            mask = [0] * length
            if pd.isna(core_str):
                return mask
            
            # Convert to string and extract digits
            cores = [int(c) for c in str(core_str) if c.isdigit()]
            
            # Set corresponding positions to 1 (1-indexed to 0-indexed)
            for core_id in cores:
                if 1 <= core_id <= length:
                    mask[core_id - 1] = 1
            
            return mask
        
        # Create binary mask columns
        masks = df['core_combination'].apply(to_binary_mask)
        
        for i in range(length):
            df[f"{prefix}_{i}"] = masks.apply(lambda x: x[i]).astype(dtype)
        
        return df
    
    def parse_cores_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse cores_str column to extract number of cores.
        
        Handles formats like:
        - '1,2,3' -> 3
        - '1' -> 1
        - '1,2,3,4,5,6,7,8' -> 8
        """
        print("Parsing cores_str")
        
        def count_cores(core_str):
            if pd.isna(core_str):
                return 0
            
            # Convert to string and count commas + 1
            core_str = str(core_str)
            if ',' in core_str:
                return len(core_str.split(','))
            elif core_str.strip():
                return 1
            else:
                return 0
        
        df['cores_str'] = df['cores_str'].apply(count_cores).astype('int8')
        
        print(f"  cores_str parsed: min={df['cores_str'].min()}, max={df['cores_str'].max()}")
        
        return df
    
    def aggregate_thermal_zones(self, df: pd.DataFrame, 
                                config: Dict[str, Any]) -> pd.DataFrame:
        """
        Aggregate thermal_zone_before* columns into single mean value.
        Excludes zeros (inactive sensors).
        """
        print("Aggregating thermal zones")
        
        prefix = config['input_prefix']
        output_col = config['output_column']
        dtype = config['dtype']
        
        # Find all thermal zone columns
        thermal_cols = [col for col in df.columns if col.startswith(prefix)]
        
        if not thermal_cols:
            print(f"Warning: No columns found with prefix '{prefix}'")
            df[output_col] = 0.0
            return df
        
        # Calculate mean excluding zeros
        def thermal_mean(row):
            values = row[thermal_cols].values
            non_zero = values[values > 0]
            return non_zero.mean() if len(non_zero) > 0 else 0.0
        
        df[output_col] = df.apply(thermal_mean, axis=1).astype(dtype)
        
        print(f"Created {output_col}: mean={df[output_col].mean():.2f}, "
              f"std={df[output_col].std():.2f}")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, 
                          config: Dict[str, Any]) -> pd.DataFrame:
        """Encode categorical variables."""
        print("Encoding categorical features")
        
        for feature, feat_config in config.items():
            if feature not in df.columns:
                print(f"Warning: Column '{feature}' not found, skipping")
                continue
            
            dtype = feat_config['dtype']
            
            if 'mapping' in feat_config:
                # Use predefined mapping
                mapping = feat_config['mapping']
                df[feature] = df[feature].map(mapping).fillna(-1).astype(dtype)
                self.label_encoders[feature] = mapping
                
            elif feat_config.get('auto_detect', False):
                # Auto-detect unique values
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
                df[feature] = df[feature].astype(dtype)
                
                # Store mapping
                self.label_encoders[feature] = dict(
                    zip(le.classes_, le.transform(le.classes_))
                )
            
            print(f"  {feature}: {len(df[feature].unique())} unique values")
        
        return df
    
    def cast_numerical_types(self, df: pd.DataFrame, 
                            config: Dict[str, Any]) -> pd.DataFrame:
        """Cast numerical columns to specified dtypes."""
        print("Casting numerical types")
        
        for feature, feat_config in config.items():
            if feature in df.columns:
                dtype = feat_config['dtype']
                df[feature] = df[feature].astype(dtype)
        
        return df
    
    def clean_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Clean data: remove duplicates, handle missing values, apply constraints."""
        print("Cleaning data")
        
        initial_rows = len(df)
        
        # Remove duplicates
        if config.get('remove_duplicates', True):
            df = df.drop_duplicates()
            print(f"  Removed {initial_rows - len(df)} duplicate rows")
        
        # Drop rows with missing values
        if config.get('drop_na', True):
            df = df.dropna()
            print(f"  Dropped {initial_rows - len(df)} rows with NaN")
        
        # Apply constraints
        if 'constraints' in config:
            for col, limits in config['constraints'].items():
                if col in df.columns:
                    before = len(df)
                    if 'min' in limits:
                        df = df[df[col] >= limits['min']]
                    if 'max' in limits:
                        df = df[df[col] <= limits['max']]
                    removed = before - len(df)
                    if removed > 0:
                        print(f"  Removed {removed} rows violating {col} constraints")
        
        print(f"Final dataset: {len(df)} rows")
        return df
    
    def normalize_data(self, df: pd.DataFrame, config: Dict[str, Any],
                      fit_scaler: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.
        
        Args:
            df: DataFrame to normalize
            config: Normalization configuration
            fit_scaler: If True, fit scaler on this data. If False, use existing scaler.
        """
        print("Normalizing data")
        
        # Get columns to normalize
        cols_to_norm = config['numerical_features'] + config['target']
        cols_to_norm = [col for col in cols_to_norm if col in df.columns]
        
        if fit_scaler:
            # Fit scaler on this data
            self.scaler = StandardScaler()
            df[cols_to_norm] = self.scaler.fit_transform(df[cols_to_norm])
            
            # Store scaler parameters
            self.scaler_params = {
                'mean': self.scaler.mean_.tolist(),
                'std': self.scaler.scale_.tolist(),
                'columns': cols_to_norm
            }
            print(f"  Fitted scaler on {len(cols_to_norm)} columns")
        else:
            # Use existing scaler
            if self.scaler is None:
                raise ValueError("Scaler not fitted yet. Set fit_scaler=True first.")
            df[cols_to_norm] = self.scaler.transform(df[cols_to_norm])
            print(f"  Applied existing scaler to {len(cols_to_norm)} columns")
        
        return df
    
    def process_dataset(self, input_path: str, output_processed: str,
                       output_normalized: str, output_metadata: str,
                       fit_scaler: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Returns:
            Tuple of (processed_df, normalized_df)
        """
        print(f"\n{'='*60}")
        print(f"Processing: {input_path}")
        print(f"{'='*60}\n")
        
        # Load data
        df = self.load_raw_data(input_path)
        
        # Parse array features
        for feature, feat_config in self.config['features']['arrays'].items():
            if feature in df.columns:
                df = self.parse_array_column(df, feature, feat_config)
        
        # Parse special features
        if 'core_combination' in df.columns:
            df = self.parse_core_combination(
                df, 
                self.config['features']['special']['core_combination']
            )
        
        # Parse cores_str
        if 'cores_str' in df.columns:
            df = self.parse_cores_str(df)
        
        # Aggregate thermal zones
        df = self.aggregate_thermal_zones(
            df, 
            self.config['features']['thermal']
        )
        
        # Encode categorical features
        df = self.encode_categorical(
            df, 
            self.config['features']['categorical']
        )
        
        # Cast numerical types
        df = self.cast_numerical_types(
            df, 
            self.config['features']['numerical']
        )
        
        # Clean data
        df = self.clean_data(df, self.config['cleaning'])
        
        # Select final columns
        final_cols = self._get_final_columns()
        df_processed = df[final_cols].copy()
        
        # Save processed data
        df_processed.to_csv(output_processed, index=False)
        print(f"\nSaved processed data: {output_processed}")
        
        # Normalize
        df_normalized = df_processed.copy()
        df_normalized = self.normalize_data(
            df_normalized, 
            self.config['normalization'],
            fit_scaler=fit_scaler
        )
        
        # Save normalized data
        df_normalized.to_csv(output_normalized, index=False)
        print(f"Saved normalized data: {output_normalized}")
        
        # Save metadata
        metadata = {
            'label_encoders': self._convert_to_json_serializable(self.label_encoders),
            'scaler_params': self._convert_to_json_serializable(self.scaler_params),
            'final_columns': final_cols,
            'num_samples': int(len(df_processed)),  # Convert to native Python int
            'gat_structure': self.config['gat_structure']
        }

        with open(output_metadata, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {output_metadata}\n")
        
        return df_processed, df_normalized

    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy/pandas types to JSON serializable Python types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _get_final_columns(self) -> List[str]:
        """Get list of final columns in correct order."""
        cols = []
        
        # Categorical
        cols.extend(self.config['features']['categorical'].keys())
        
        # Numerical
        cols.extend(self.config['features']['numerical'].keys())
        
        # Thermal aggregation
        cols.append(self.config['features']['thermal']['output_column'])
        
        # Array features (expanded)
        for feat_config in self.config['features']['arrays'].values():
            prefix = feat_config['output_prefix']
            length = feat_config['length']
            cols.extend([f"{prefix}_{i}" for i in range(length)])
        
        # Special features (expanded)
        special_config = self.config['features']['special']['core_combination']
        prefix = special_config['output_prefix']
        length = special_config['length']
        cols.extend([f"{prefix}_{i}" for i in range(length)])
        
        # Targets
        for target in self.config['targets']:
            cols.append(target['name'])
        
        return cols


def merge_datasets(processed_paths: List[str], normalized_paths: List[str],
                   output_processed: str, output_normalized: str) -> None:
    """
    Merge multiple processed datasets.
    
    Args:
        processed_paths: List of processed CSV paths
        normalized_paths: List of normalized CSV paths
        output_processed: Output path for merged processed data
        output_normalized: Output path for merged normalized data
    """
    print(f"\n{'='*60}")
    print("Merging datasets")
    print(f"{'='*60}\n")
    
    # Merge processed
    dfs_processed = [pd.read_csv(p) for p in processed_paths]
    df_merged_processed = pd.concat(dfs_processed, ignore_index=True)
    df_merged_processed.to_csv(output_processed, index=False)
    print(f"Merged processed: {len(df_merged_processed)} rows -> {output_processed}")
    
    # Merge normalized
    dfs_normalized = [pd.read_csv(p) for p in normalized_paths]
    df_merged_normalized = pd.concat(dfs_normalized, ignore_index=True)
    df_merged_normalized.to_csv(output_normalized, index=False)
    print(f"Merged normalized: {len(df_merged_normalized)} rows -> {output_normalized}\n")