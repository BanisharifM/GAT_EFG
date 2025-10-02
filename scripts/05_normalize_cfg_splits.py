#!/usr/bin/env python3
"""Normalize the CFG-augmented log-transformed splits."""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

# Load the CFG-augmented splits
train = pd.read_csv('data/splits/train_log_with_cfg.csv')
val = pd.read_csv('data/splits/val_log_with_cfg.csv')
test = pd.read_csv('data/splits/test_log_with_cfg.csv')

# Get all CFG columns
cfg_cols = [col for col in train.columns if col.startswith('cfg_')]

# Columns to normalize (original + CFG features + time_elapsed_log)
cols_to_normalize = [
    'frequency_index', 'num_cores', 'cores_str', 'avg_temp_before', 
    'thermal_init_mean', 
    'freq_lvl_0', 'freq_lvl_1', 'freq_lvl_2', 'freq_lvl_3', 
    'freq_lvl_4', 'freq_lvl_5', 'freq_lvl_6', 'freq_lvl_7',
    'freq_hz_0', 'freq_hz_1', 'freq_hz_2', 'freq_hz_3', 
    'freq_hz_4', 'freq_hz_5', 'freq_hz_6', 'freq_hz_7',
    'time_elapsed_log'  # Log-transformed target
] + cfg_cols  # Add all CFG features

print(f"Normalizing {len(cols_to_normalize)} features")
print(f"  Original features: {len(cols_to_normalize) - len(cfg_cols)}")
print(f"  CFG features: {len(cfg_cols)}")

# Fit scaler on training data
scaler = StandardScaler()
train[cols_to_normalize] = scaler.fit_transform(train[cols_to_normalize])
val[cols_to_normalize] = scaler.transform(val[cols_to_normalize])
test[cols_to_normalize] = scaler.transform(test[cols_to_normalize])

# Save normalized splits (overwrite)
train.to_csv('data/splits/train_log_with_cfg.csv', index=False)
val.to_csv('data/splits/val_log_with_cfg.csv', index=False)
test.to_csv('data/splits/test_log_with_cfg.csv', index=False)

# Save scaler params
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'std': scaler.scale_.tolist(),
    'columns': cols_to_normalize,
    'n_samples': len(train)
}

with open('data/processed/scaler_params_log_with_cfg.json', 'w') as f:
    json.dump({'scaler_params': scaler_params}, f, indent=2)

print("\n✓ Normalized CFG-augmented log-transformed splits")
print(f"✓ Scaler params saved to: data/processed/scaler_params_log_with_cfg.json")
