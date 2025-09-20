import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the raw train and test CSV files from the specified directory.

    Args:
        data_dir (Path): The directory containing 'train.csv' and 'test.csv'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test DataFrames.
    """
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        logging.error(f"Data files not found in {data_dir}. Please run `make setup` first.")
        raise FileNotFoundError(f"Data files not found in {data_dir}")

    logging.info(f"Loading data from {data_dir}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logging.info("Data loaded successfully.")
    
    return train_df, test_df

def save_processed(df: pd.DataFrame, path: Path):
    """
    Saves a DataFrame to a specified path, creating the directory if needed.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (Path): The output file path (e.g., 'data/processed/train.csv').
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Saved processed data to {path}")