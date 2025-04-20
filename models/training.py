import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from monitoring.logger import setup_logger
from models.baseline import NaiveForecaster

# Set up logger
logger = setup_logger("model_training", "logs/training.log")

def create_directory_if_not_exists(directory_path: str) -> None:
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features for time series forecasting
    
    Args:
        df: Input dataframe with time series data
        
    Returns:
        Tuple of (X, y) where X contains features and y contains target values
    """
    # Create lagged features
    for lag in range(1, 13):  # Create lags up to 12 months
        df[f'lag_{lag}'] = df.groupby('country_code')['value'].shift(lag)
    
    # Create rolling statistics
    for window in [3, 6, 12]:
        df[f'rolling_mean_{window}'] = df.groupby('country_code')['value'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}'] = df.groupby('country_code')['value'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # Create seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop rows with NaN (caused by lagged features)
    df = df.dropna()
    
    # Select features and target
    features = [col for col in df.columns if col.startswith('lag_') or 
                col.startswith('rolling_') or 
                col in ['month_sin', 'month_cos', 'year', 'month', 'quarter']]
    
    X = df[features]
    y = df['value']
    
    return X, y

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate a model using multiple metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred))
    }
    
    return metrics

def train_and_compare_models(data_path: str, output_dir: str = "models/model_repository") -> Dict[str, Any]:
    """
    Train and compare multiple models
    
    Args:
        data_path: Path to the processed data
        output_dir: Directory to save trained models
        
    Returns:
        Dictionary with model evaluation results
    """
    logger.info(f"Training models using data from {data_path}")
    
    # Create output directory if it doesn't exist
    create_directory_if_not_exists(output_dir)
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date to ensure proper time series splitting
        df = df.sort_values(by=['country_code', 'date'])
        
        # Prepare features
        X, y = prepare_features(df)
        
        # Split data into training and testing sets using time-based split
        # Get the timestamp cutoff for the last 20% of the time range
        timestamps = df['date'].unique()
        timestamps.sort()
        split_idx = int(len(timestamps) * 0.8)
        split_date = timestamps[split_idx]
        
        train_mask = df['date'] < split_date
        test_mask = df['date'] >= split_date
        
        X_train = X[train_mask.values]
        y_train = y[train_mask.values]
        X_test = X[test_mask.values]
        y_test = y[test_mask.values]
        
        logger.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
        
        # Define models to train
        models = {
            'baseline': NaiveForecaster(),
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf')
        }
        
        # Train and evaluate each model
        results = {}
        best_model = None
        best_metric = float('inf')  # Lower RMSE is better
        
        for name, model in models.items():
            logger.info(f"Training {name} model")
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Evaluate the model
            metrics = evaluate_model(model, X_test, y_test)
            logger.info(f"{name} model metrics: {metrics}")
            
            # Save model and metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{name}_model_{timestamp}.pkl"
            model_path = os.path.join(output_dir, model_filename)
            
            # Save the model
            joblib.dump(model, model_path)
            
            # Save model info
            model_info = {
                'name': name,
                'filename': model_filename,
                'metrics': metrics,
                'timestamp': timestamp,
                'version': f"{name}_v1.0_{timestamp}",
                'type': name
            }
            
            model_info_path = os.path.join(output_dir, f"{name}_info_{timestamp}.json")
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            
            results[name] = model_info
            
            # Check if this is the best model
            if metrics['rmse'] < best_metric:
                best_metric = metrics['rmse']
                best_model = model
                best_model_info = model_info
        
        # Save the best model as "best_model.pkl"
        best_model_path = os.path.join(output_dir, "best_model.pkl")
        joblib.dump(best_model, best_model_path)
        
        # Save best model info
        best_model_info_path = os.path.join(output_dir, "best_model_info.json")
        with open(best_model_info_path, 'w') as f:
            json.dump(best_model_info, f, indent=4)
        
        logger.info(f"Best model: {best_model_info['name']} with RMSE: {best_metric}")
        logger.info(f"Models trained and saved to {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument("--data", required=True, help="Path to processed data file")
    parser.add_argument("--output-dir", default="models/model_repository", help="Output directory for models")
    
    args = parser.parse_args()
    
    try:
        results = train_and_compare_models(args.data, args.output_dir)
        print("Model training completed successfully.")
        print("Model results:")
        for name, info in results.items():
            print(f"{name}: RMSE={info['metrics']['rmse']:.4f}, MAE={info['metrics']['mae']:.4f}, R²={info['metrics']['r2']:.4f}")
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
