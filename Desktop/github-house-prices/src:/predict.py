import argparse
import pandas as pd
from pathlib import Path
import yaml
import logging

from data_utils import load_raw
from features import build_features
from models import train_linear_regression, train_xgboost, train_nn, save_model
from evaluate import evaluate_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RANDOM_STATE = config['base']['random_state']

def main(model_to_train: str, output_dir: Path, demo_mode: bool):
    """
    Main training script.

    Args:
        model_to_train (str): Which model to train ('baseline', 'xgb', 'nn', or 'all').
        output_dir (Path): Directory to save models and metrics.
        demo_mode (bool): If True, trains on a small subset of data for quick tests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting training process for model(s): {model_to_train}")
    logging.info(f"Output will be saved to: {output_dir}")

    # 1. Load Data
    train_df, _ = load_raw(ROOT_DIR / "data/raw")
    if demo_mode:
        logging.warning("DEMO MODE: Using only the first 200 rows of data.")
        train_df = train_df.head(200)

    # 2. Build Features
    X, y, preprocessor = build_features(train_df, fit_mode=True)
    logging.info(f"Feature matrix shape: {X.shape}")
    
    # Save the preprocessor
    save_model(preprocessor, output_dir / "preprocessor.joblib")

    # 3. Train Models
    metrics = {}

    if model_to_train in ['baseline', 'all']:
        model, cv_rmse = train_linear_regression(X, y)
        save_model(model, output_dir / "linear_regression.joblib")
        metrics['LinearRegression'] = {'CV_RMSE_log': cv_rmse}

    if model_to_train in ['xgb', 'all']:
        params = config['models']['xgboost']['params']
        model, cv_rmse = train_xgboost(X, y, params)
        save_model(model, output_dir / "xgboost.joblib")
        metrics['XGBoost'] = {'CV_RMSE_log': cv_rmse}

    if model_to_train in ['nn', 'all']:
        nn_config = config['models']['neural_network']
        model, val_rmse = train_nn(X, y, nn_config)
        save_model(model, output_dir / "neural_network.keras")
        metrics['NeuralNetwork'] = {'CV_RMSE_log': val_rmse}

    # 4. Save Metrics
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path)
    logging.info(f"Metrics saved to {metrics_path}")
    print("\n--- Training Metrics ---")
    print(metrics_df)
    print("----------------------\n")

    logging.info("Training process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train house price prediction models.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["baseline", "xgb", "nn", "all"],
        help="The model to train."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=ROOT_DIR / "models",
        help="Directory to save model artifacts."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode on a small subset of data."
    )
    args = parser.parse_args()
    main(args.model, args.out_dir, args.demo)