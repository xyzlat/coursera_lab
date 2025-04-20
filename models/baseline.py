import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class NaiveForecaster(BaseEstimator, RegressorMixin):
    """
    A simple naive forecasting model that predicts the next value
    based on the last observed value (persistence model)
    """
    
    def __init__(self):
        self.is_fitted_ = False
        self.lag_feature_name = None
    
    def fit(self, X, y):
        """
        Fit the model by identifying which feature is the most recent lag
        
        Args:
            X: Features dataframe
            y: Target values
        
        Returns:
            self: The fitted model
        """
        # Find the lag_1 feature in X
        lag_features = [col for col in X.columns if col.startswith('lag_')]
        
        if not lag_features:
            raise ValueError("No lag features found in input data. Expected at least 'lag_1'")
        
        # Sort the lag features to get the one with the lowest lag (most recent)
        lag_features.sort(key=lambda x: int(x.split('_')[1]))
        self.lag_feature_name = lag_features[0]  # Should be 'lag_1'
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions by using the most recent lag value
        
        Args:
            X: Features dataframe
        
        Returns:
            np.array: Predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        
        if self.lag_feature_name not in X.columns:
            raise ValueError(f"Required feature {self.lag_feature_name} not found in input data")
        
        # Simply return the most recent lag value as the prediction
        return X[self.lag_feature_name].values

class SeasonalNaiveForecaster(BaseEstimator, RegressorMixin):
    """
    A seasonal naive forecasting model that predicts based on the value
    from the same season in the previous period (e.g., same month last year)
    """
    
    def __init__(self, seasonal_period=12):
        self.seasonal_period = seasonal_period
        self.is_fitted_ = False
        self.seasonal_lag_name = None
    
    def fit(self, X, y):
        """
        Fit the model by identifying which feature corresponds to the seasonal lag
        
        Args:
            X: Features dataframe
            y: Target values
        
        Returns:
            self: The fitted model
        """
        # Find the appropriate seasonal lag feature
        expected_lag_name = f'lag_{self.seasonal_period}'
        if expected_lag_name not in X.columns:
            raise ValueError(f"Required seasonal lag feature {expected_lag_name} not found in input data")
        
        self.seasonal_lag_name = expected_lag_name
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the seasonal lag value
        
        Args:
            X: Features dataframe
        
        Returns:
            np.array: Predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        
        if self.seasonal_lag_name not in X.columns:
            raise ValueError(f"Required feature {self.seasonal_lag_name} not found in input data")
        
        # Return the seasonal lag value as the prediction
        return X[self.seasonal_lag_name].values

class AverageForecaster(BaseEstimator, RegressorMixin):
    """
    A simple forecasting model that predicts the next value
    based on the average of historical values
    """
    
    def __init__(self, window=12):
        self.window = window
        self.is_fitted_ = False
        self.rolling_mean_name = None
    
    def fit(self, X, y):
        """
        Fit the model by identifying which feature is the rolling mean
        
        Args:
            X: Features dataframe
            y: Target values
        
        Returns:
            self: The fitted model
        """
        # Find the rolling mean feature in X
        expected_feature = f'rolling_mean_{self.window}'
        if expected_feature not in X.columns:
            available_means = [col for col in X.columns if col.startswith('rolling_mean_')]
            if not available_means:
                raise ValueError("No rolling mean features found in input data")
            
            # Use the first available rolling mean
            self.rolling_mean_name = available_means[0]
        else:
            self.rolling_mean_name = expected_feature
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions by using the rolling mean value
        
        Args:
            X: Features dataframe
        
        Returns:
            np.array: Predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        
        if self.rolling_mean_name not in X.columns:
            raise ValueError(f"Required feature {self.rolling_mean_name} not found in input data")
        
        # Return the rolling mean value as the prediction
        return X[self.rolling_mean_name].values
