# Process all datasets (rubikpi, tx2, nx) and merge
python scripts/preprocess_data.py --config configs/preprocess_config.yaml

# Process only RubikPi dataset
python scripts/preprocess_data.py --config configs/preprocess_config.yaml --dataset rubikpi

# Process only TX2 dataset
python scripts/preprocess_data.py --config configs/preprocess_config.yaml --dataset tx2