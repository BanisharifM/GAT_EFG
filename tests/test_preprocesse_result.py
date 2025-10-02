import pandas as pd
import json

# Load normalized datasets
df_rubikpi = pd.read_csv('data/processed/rubikpi_normalized.csv')
df_tx2 = pd.read_csv('data/processed/tx2_normalized.csv')
df_merged = pd.read_csv('data/processed/merged_normalized.csv')

# Load metadata
with open('data/processed/merged_metadata.json', 'r') as f:
    metadata = json.load(f)

print("="*60)
print("NORMALIZED DATA VERIFICATION")
print("="*60)

# Check row counts
print(f"\nRow counts:")
print(f"  RubikPi:  {len(df_rubikpi):,}")
print(f"  TX2:      {len(df_tx2):,}")
print(f"  Merged:   {len(df_merged):,}")
print(f"  Total:    {len(df_rubikpi) + len(df_tx2):,}")

# Check normalization (should be ~0 mean, ~1 std for merged)
print(f"\nNormalization check (merged data):")
norm_cols = ['freq_hz_0', 'thermal_init_mean', 'time_elapsed']
for col in norm_cols:
    print(f"  {col:20s} → mean: {df_merged[col].mean():7.4f}, std: {df_merged[col].std():7.4f}")

# Check binary features (should NOT be normalized)
print(f"\nBinary features (should be 0/1):")
print(f"  run_mode:      {df_merged['run_mode'].unique()}")
print(f"  core_active_0: {df_merged['core_active_0'].unique()}")

# Check platform distribution
print(f"\nPlatform distribution:")
print(df_merged['platform'].value_counts())

# Check scaler info
print(f"\nScaler fitted on:")
print(f"  Samples: {metadata['scaler_params']['n_samples']:,}")
print(f"  Columns: {len(metadata['scaler_params']['columns'])}")
print(f"\nExample scaler params:")
print(f"  Mean (first 3): {metadata['scaler_params']['mean'][:3]}")
print(f"  Std  (first 3): {metadata['scaler_params']['std'][:3]}")

print("\n" + "="*60)