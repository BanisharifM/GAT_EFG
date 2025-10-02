import pandas as pd
import numpy as np

# Load splits
train = pd.read_csv('data/splits/train.csv')
val = pd.read_csv('data/splits/val.csv')
test = pd.read_csv('data/splits/test.csv')

# Create unique identifiers from row content (hash of all values)
def create_row_hash(df):
    """Create unique hash for each row based on content."""
    return df.apply(lambda row: hash(tuple(row)), axis=1)

train_hashes = set(create_row_hash(train))
val_hashes = set(create_row_hash(val))
test_hashes = set(create_row_hash(test))

# Check for overlaps
train_val_overlap = train_hashes & val_hashes
train_test_overlap = train_hashes & test_hashes
val_test_overlap = val_hashes & test_hashes

print("Data Leakage Check:")
print(f"  Train/Val overlap:  {len(train_val_overlap)} rows")
print(f"  Train/Test overlap: {len(train_test_overlap)} rows")
print(f"  Val/Test overlap:   {len(val_test_overlap)} rows")

if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
    print("\n✅ No data leakage - splits are properly disjoint!")
else:
    print("\n❌ Data leakage detected!")

# Verify total count
total = len(train) + len(val) + len(test)
print(f"\nTotal samples: {total} (should be 4,704)")

# Check platform balance
print("\nPlatform balance:")
for name, df in [('Train', train), ('Val', val), ('Test', test)]:
    platform_pct = df['platform'].value_counts(normalize=True) * 100
    rubikpi_pct = platform_pct.get(0, 0)
    tx2_pct = platform_pct.get(1, 0)
    print(f"  {name:5s}: RubikPi={rubikpi_pct:.1f}%, TX2={tx2_pct:.1f}%")

# Check all benchmarks present
print(f"\nBenchmark coverage:")
all_benchmarks = set(train['benchmark']) | set(val['benchmark']) | set(test['benchmark'])
print(f"  Total unique: {len(all_benchmarks)}")
print(f"  Train: {len(set(train['benchmark']))} benchmarks")
print(f"  Val:   {len(set(val['benchmark']))} benchmarks")
print(f"  Test:  {len(set(test['benchmark']))} benchmarks")