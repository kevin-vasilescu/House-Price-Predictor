import joblib
from pathlib import Path
import logging

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Saving and Loading ---

def save_model(obj, path: Path):
    """Saves a model object to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, tf.keras.Model):
        obj.save(path)
    else:
        joblib.dump(obj, path)
    logging.info(f"Model saved to {path}")

def load_model(path: Path):
    """Loads a model object from a file."""
    if not path.exists():
        logging.error(f"Model file not found at {path}")
        raise FileNotFoundError(f"No model file at {path}")
    
    if path.suffix in ['.h5', '.keras']:
        model = tf.keras.models.load_model(path)
    else:
        model = joblib.load(path)
    logging.info(f"Model loaded from {path}")
    return model

# --- Model Training Functions ---

def train_linear_regression(X, y, cv_folds=5):
    """Trains a Linear Regression model with cross-validation."""
    logging.info("Training Linear Regression model...")
    model = LinearRegression()
    
    # Use negative mean squared error because sklearn cross_val_score maximizes a utility function
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
    rmse_scores = -scores
    logging.info(f"Cross-validation RMSE scores: {rmse_scores}")
    logging.info(f"Mean CV RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
    
    model.fit(X, y)
    return model, rmse_scores.mean()

def train_xgboost(X, y, params, cv_folds=5):
    """Trains an XGBoost model."""
    logging.info("Training XGBoost model...")
    model = xgb.XGBRegressor(**params)
    
    # XGBoost doesn't have a direct CV training method like scikit-learn,
    # but we can use cross_val_score for evaluation. The final model is trained on all data.
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
    rmse_scores = -scores
    logging.info(f"Cross-validation RMSE scores: {rmse_scores}")
    logging.info(f"Mean CV RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")

    model.fit(X, y) # Train final model on all data
    return model, rmse_scores.mean()

def train_nn(X, y, config, validation_split=0.2):
    """Trains a Keras Neural Network model."""
    logging.info("Training Neural Network model...")
    
    # Keras needs numpy arrays
    X_np = X.toarray() if hasattr(X, "toarray") else np.array(X)
    y_np = y.values if isinstance(y, pd.Series) else y

    model = Sequential([
        Dense(config['layers'][0], activation='relu', input_shape=(X_np.shape[1],)),
        Dropout(config['dropout_rates'][0]),
        Dense(config['layers'][1], activation='relu'),
        Dropout(config['dropout_rates'][1]),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['patience'],
        restore_best_weights=True
    )
    
    history = model.fit(
        X_np, y_np,
        validation_split=validation_split,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )
    
    # The returned validation RMSE is on the log-transformed target
    val_rmse = min(history.history['val_root_mean_squared_error'])
    logging.info(f"Best Validation (log) RMSE: {val_rmse:.4f}")

    return model, val_rmse