# Step 1: Process all raw files
python scripts/01_process_data.py --config configs/preprocess_config.yaml

# Step 2: Normalize each dataset independently
python scripts/02_normalize_data.py --config configs/preprocess_config.yaml


# Step 1: Process all raw files + create merged
python scripts/01_process_data.py --config configs/preprocess_config.yaml

# Step 2: Fit scaler on merged, normalize all
python scripts/02_normalize_data.py --config configs/preprocess_config.yaml --fit-on-merged


# Process one file
python scripts/01_process_data.py --config configs/preprocess_config.yaml

# Normalize just rubikpi
python scripts/02_normalize_data.py \
    --input data/processed/rubikpi_processed.csv \
    --output data/processed/rubikpi_normalized.csv

    
    
python scripts/03_split_data.py --input data/processed/merged_normalized.csv
python scripts/03_split_data.py --input data/processed/merged_normalized.csv --ratios 0.8 0.1 0.1
python scripts/03_split_data.py --input data/processed/merged_normalized.csv --seed 123