#!/usr/bin/env python3
"""Create train/val/test splits with log-transformed target from scratch."""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load PROCESSED (unnormalized) data
df = pd.read_csv('data/processed/merged_processed.csv')

print(f"Loaded: {len(df)} rows")
print(f"Original time_elapsed: min={df['time_elapsed'].min():.4f}, max={df['time_elapsed'].max():.4f}")

# Add log transform
df['time_elapsed_log'] = np.log1p(df['time_elapsed'])
print(f"Log-transformed: min={df['time_elapsed_log'].min():.4f}, max={df['time_elapsed_log'].max():.4f}")

# Stratified split
strat_key = df['platform'].astype(str) + '_' + df['benchmark'].astype(str)

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=strat_key, random_state=42)
temp_strat = temp_df['platform'].astype(str) + '_' + temp_df['benchmark'].astype(str)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_strat, random_state=42)

print(f"\nSplits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# Normalize
cols_to_norm = [
    'frequency_index', 'num_cores', 'cores_str', 'avg_temp_before', 'thermal_init_mean',
    'freq_lvl_0', 'freq_lvl_1', 'freq_lvl_2', 'freq_lvl_3', 'freq_lvl_4', 'freq_lvl_5', 'freq_lvl_6', 'freq_lvl_7',
    'freq_hz_0', 'freq_hz_1', 'freq_hz_2', 'freq_hz_3', 'freq_hz_4', 'freq_hz_5', 'freq_hz_6', 'freq_hz_7',
    'time_elapsed_log'
]

scaler = StandardScaler()
train_df[cols_to_norm] = scaler.fit_transform(train_df[cols_to_norm])
val_df[cols_to_norm] = scaler.transform(val_df[cols_to_norm])
test_df[cols_to_norm] = scaler.transform(test_df[cols_to_norm])

# Save
train_df.to_csv('data/splits/train_log.csv', index=False)
val_df.to_csv('data/splits/val_log.csv', index=False)
test_df.to_csv('data/splits/test_log.csv', index=False)

# Save scaler
with open('data/processed/scaler_params_log.json', 'w') as f:
    json.dump({
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
            'columns': cols_to_norm,
            'n_samples': len(train_df)
        }
    }, f, indent=2)

print("\nSaved splits with log-transformed targets")
print(f"Normalized time_elapsed_log: mean={train_df['time_elapsed_log'].mean():.4f}, std={train_df['time_elapsed_log'].std():.4f}")
