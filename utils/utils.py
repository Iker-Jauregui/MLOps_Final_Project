import pandas as pd
from pathlib import Path

def cargar_datos(data_folder_name: str = 'data', filename: str = 'ensemble_2022-07-01_2025-10-01') -> pd.DataFrame:
    # 1. Route configuration (we use pathlib for better path handling)
    # If you're in a .py script use: Path(__file__).parent
    # If you're in a Notebook use: Path.cwd()
    base_path = Path.cwd().parent 
    data_folder = base_path / data_folder_name
    
    parquet_path = data_folder / f'{filename}.parquet'
    csv_path = data_folder / f'{filename}.csv' 

    # 2. Search for Parquet first, then CSV
    if parquet_path.exists():
        print(f"📦 Loading data from Parquet: {parquet_path.name}")
        return pd.read_parquet(parquet_path)
    
    elif csv_path.exists():
        print(f"📄 Loading data from CSV: {csv_path.name}")
        # We use sep=';' 
        df = pd.read_csv(csv_path, sep=';')
        df.to_parquet(parquet_path, index=False)  # Convert and save as Parquet
        print(f"✅ Data converted and saved as Parquet: {parquet_path.name}")
        return df
    
    else:
        # 3. Raise exception if nothing is found
        raise FileNotFoundError(
            f"❌ No data file found in {data_folder}.\n"
            f"Searched for: '{parquet_path.name}' and '{csv_path.name}'"
        )
