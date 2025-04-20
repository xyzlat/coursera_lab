import pandas as pd
import numpy as np
import os
import joblib
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from monitoring.logger import setup_logger

# Set up logger
logger = setup_logger("model_evaluation", "logs/evaluation.log")

def load_model_and_info(model_path: str = "models/model_repository/best_model.pkl") -> Tuple[Any, Dict[str, Any]]:
    """
    Load a trained model and its information
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        Tuple of (model, model_info)
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Load model info from the corresponding JSON file
    model_info_path = model_path.replace(".pkl", "_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
    else:
        # If model info doesn't exist, create minimal info
        model_info = {
            "name": os.path.basename(model_path).split('.')[0],
            "version": "unknown",
            "type": "unknown",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    
    return model, model_info

def prepare_prediction_features(country: str, horizon: int, 
                               data_path: str = "data/processed") -> pd.DataFrame:
    """
    Prepare features for prediction for a specific country
    
    Args:
        country: Country code to prepare features for
        horizon: Number of periods to forecast
        data_path: Path to the processed data directory
        
    Returns:
        DataFrame with features for prediction
    """
    # Find the most recent processed data file
    data_files = [f for f in os.listdir(data_path) if f.startswith("processed_data_")]
    data_files.sort(reverse=True)  # Sort in descending order to get the most recent first
    
    if not data_files:
        raise ValueError(f"No processed data files found in {data_path}")
    
    latest_data_file = os.path.join(data_path, data_files[0])
    
    # Load the data
    df = pd.read_csv(latest_data_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for the specified country
    country = country.upper()
    country_data = df[df['country_code'] == country]
    
    if country_data.empty:
        raise ValueError(f"No data found for country code: {country}")
    
    # Sort by date to get the most recent data first
    country_data = country_data.sort_values(by='date', ascending=False)
    
    # Get the most recent date in the data
    last_date = country_data['date'].max()
    
    # Create features for future periods
    future_dates = [last_date + timedelta(days=30*i) for i in range(1, horizon+1)]
    future_df = pd.DataFrame({
        'date': future_dates,
        'country_code': country,
        'year': [d.year for d in future_dates],
        'month': [d.month for d in future_dates],
        'quarter': [((d.month-1) // 3) + 1 for d in future_dates],
        'month_sin': [np.sin(2 * np.pi * d.month / 12) for d in future_dates],
        'month_cos': [np.cos(2 * np.pi * d.month / 12) for d in future_dates]
    })
    
    # Get the most recent values for lagged features
    most_recent_values = country_data.head(12)['value'].tolist()
    
    # Fill with the last available value if we don't have enough history
    while len(most_recent_values) < 12:
        most_recent_values.append(most_recent_values[-1])
    
    # Create lagged features for the forecast periods
    for i in range(horizon):
        row_lags = {}
        for lag in range(1, 13):
            if i < lag:
                # Use historical data for the first forecast steps
                row_lags[f'lag_{lag}'] = most_recent_values[lag-1-i]
            else:
                # Use previous forecasts
                row_lags[f'lag_{lag}'] = future_df.loc[i-lag, 'value'] if 'value' in future_df else most_recent_values[0]
        
        for col, val in row_lags.items():
            future_df.loc[i, col] = val
    
    # Calculate rolling statistics for the forecast periods
    for window in [3, 6, 12]:
        # For the first forecast steps, use historical data
        historical_values = most_recent_values[:window]
        for i in range(horizon):
            if i == 0:
                rolling_values = historical_values
            elif i < window:
                rolling_values = [future_df.loc[j, 'value'] if 'value' in future_df else most_recent_values[0] 
                                  for j in range(i)] + historical_values[:(window-i)]
            else:
                rolling_values = [future_df.loc[j, 'value'] if 'value' in future_df else most_recent_values[0] 
                                  for j in range(i-window, i)]
                
            future_df.loc[i, f'rolling_mean_{window}'] = np.mean(rolling_values)
            future_df.loc[i, f'rolling_std_{window}'] = np.std(rolling_values) if len(rolling_values) > 1 else 0
    
    return future_df

def predict_for_country(country: str, horizon: int, model=None, 
                        model_path: str = "models/model_repository/best_model.pkl") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate predictions for a specific country
    
    Args:
        country: Country code to predict for
        horizon: Number of periods to forecast
        model: Pre-loaded model (optional)
        model_path: Path to the model if not pre-loaded
        
    Returns:
        Tuple of (predictions, model_info)
    """
    logger.info(f"Generating predictions for country: {country}, horizon: {horizon}")
    
    try:
        # Load model if not provided
        if model is None:
            model, model_info = load_model_and_info(model_path)
        else:
            _, model_info = load_model_and_info(model_path)
        
        # Prepare features for prediction
        features_df = prepare_prediction_features(country, horizon)
        
        # Make predictions
        predictions = []
        
        # Generate predictions one step at a time
        for i in range(horizon):
            # Select features for the current time step
            current_features = features_df.iloc[[i]]
            
            # Select only the columns that the model was trained on
            feature_cols = [col for col in current_features.columns if col.startswith('lag_') or 
                            col.startswith('rolling_') or 
                            col in ['month_sin', 'month_cos', 'year', 'month', 'quarter']]
            
            X = current_features[feature_cols]
            
            # Predict
            pred_value = float(model.predict(X)[0])
            
            # Add the prediction to our features dataframe for the next steps
            features_df.loc[i, 'value'] = pred_value
            
            # Format the prediction for output
            pred_date = current_features['date'].iloc[0]
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'value': pred_value,
                'year': int(current_features['year'].iloc[0]),
                'month': int(current_features['month'].iloc[0]),
                'quarter': int(current_features['quarter'].iloc[0])
            })
        
        logger.info(f"Successfully generated {len(predictions)} predictions for {country}")
        return predictions, model_info
        
    except Exception as e:
        logger.error(f"Error generating predictions for {country}: {str(e)}")
        raise

def evaluate_model_performance(model_path: str = "models/model_repository/best_model.pkl", 
                               data_path: str = "data/processed") -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance on the latest data
    
    Args:
        model_path: Path to the model to evaluate
        data_path: Path to the processed data directory
        
    Returns:
        Dictionary with evaluation metrics by country
    """
    logger.info(f"Evaluating model performance: {model_path}")
    
    try:
        # Load model
        model, _ = load_model_and_info(model_path)
        
        # Find the most recent processed data file
        data_files = [f for f in os.listdir(data_path) if f.startswith("processed_data_")]
        data_files.sort(reverse=True)
        
        if not data_files:
            raise ValueError(f"No processed data files found in {data_path}")
        
        latest_data_file = os.path.join(data_path, data_files[0])
        
        # Load data
        df = pd.read_csv(latest_data_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values(by=['country_code', 'date'])
        
        # Prepare features for evaluation
        # (similar to what's done in training.py)
        # Create lagged features
        for lag in range(1, 13):
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
        
        # Select features
        features = [col for col in df.columns if col.startswith('lag_') or 
                   col.startswith('rolling_') or 
                   col in ['month_sin', 'month_cos', 'year', 'month', 'quarter']]
        
        X = df[features]
        y = df['value']
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate overall metrics
        overall_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'mae': float(mean_absolute_error(y, y_pred)),
            'r2': float(r2_score(y, y_pred))
        }
        
        # Calculate metrics by country
        countries = df['country_code'].unique()
        country_metrics = {}
        
        for country in countries:
            country_mask = df['country_code'] == country
            country_y = y[country_mask]
            country_pred = y_pred[country_mask]
            
            country_metrics[country] = {
                'rmse': float(np.sqrt(mean_squared_error(country_y, country_pred))),
                'mae': float(mean_absolute_error(country_y, country_pred)),
                'r2': float(r2_score(country_y, country_pred)),
                'sample_size': int(country_mask.sum())
            }
        
        # Combine results
        results = {
            'overall': overall_metrics,
            'by_country': country_metrics
        }
        
        # Log the results
        logger.info(f"Overall model performance: RMSE={overall_metrics['rmse']:.4f}, MAE={overall_metrics['mae']:.4f}, R²={overall_metrics['r2']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating model performance: {str(e)}")
        raise
