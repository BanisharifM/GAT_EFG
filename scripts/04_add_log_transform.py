import pandas as pd
import numpy as np

for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'data/splits/{split}.csv')
    
    # Add log-transformed target
    df['time_elapsed_log'] = np.log1p(df['time_elapsed'])
    
    # Save
    df.to_csv(f'data/splits/{split}_log.csv', index=False)
    print(f"Added log transform to {split}.csv")
