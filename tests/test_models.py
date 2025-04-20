import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import joblib

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from models.training import prepare_features, evaluate_model, train_and_compare_models
from models.evaluation import predict_for_country, prepare_prediction_features
from models.baseline import NaiveForecaster, SeasonalNaiveForecaster, AverageForecaster

class TestBaselineModels(unittest.TestCase):
    """Tests for the baseline forecast models"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample data for testing
        np.random.seed(42)
        
        # Create features with lag columns
        dates = pd.date_range(start='2020-01-01', periods=24, freq='MS')
        self.X = pd.DataFrame({
            'lag_1': np.random.randn(24) + 100,
            'lag_2': np.random.randn(24) + 100,
            'lag_12': np.random.randn(24) + 100,
            'rolling_mean_3': np.random.randn(24) + 100,
            'rolling_mean_6': np.random.randn(24) + 100,
            'rolling_mean_12': np.random.randn(24) + 100,
        })
        
        # Target values
        self.y = np.random.randn(24) + 100
    
    def test_naive_forecaster(self):
        """Test the naive forecaster model"""
        # Create and fit the model
        model = NaiveForecaster()
        model.fit(self.X, self.y)
        
        # Check if the model identified the correct lag feature
        self.assertEqual(model.lag_feature_name, 'lag_1')
        
        # Test predictions
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        
        # Predictions should match the lag_1 values
        np.testing.assert_array_equal(predictions, self.X['lag_1'].values)
    
    def test_seasonal_naive_forecaster(self):
        """Test the seasonal naive forecaster model"""
        # Create and fit the model
        model = SeasonalNaiveForecaster(seasonal_period=12)
        model.fit(self.X, self.y)
        
        # Check if the model identified the correct seasonal lag feature
        self.assertEqual(model.seasonal_lag_name, 'lag_12')
        
        # Test predictions
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        
        # Predictions should match the lag_12 values
        np.testing.assert_array_equal(predictions, self.X['lag_12'].values)
    
    def test_average_forecaster(self):
        """Test the average forecaster model"""
        # Create and fit the model
        model = AverageForecaster(window=3)
        model.fit(self.X, self.y)
        
        # Check if the model identified the correct rolling mean feature
        self.assertEqual(model.rolling_mean_name, 'rolling_mean_3')
        
        # Test predictions
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        
        # Predictions should match the rolling_mean_3 values
        np.testing.assert_array_equal(predictions, self.X['rolling_mean_3'].values)
    
    def test_model_error_handling(self):
        """Test error handling in the models"""
        # Test with missing lag feature
        X_missing = self.X.drop(columns=['lag_1'])
        model = NaiveForecaster()
        
        # Should raise error when fitting
        with self.assertRaises(ValueError):
            model.fit(X_missing, self.y)
        
        # Test prediction without fitting
        model = SeasonalNaiveForecaster()
        with self.assertRaises(ValueError):
            model.predict(self.X)
        
        # Test prediction with missing feature
        model = AverageForecaster()
        model.fit(self.X, self.y)
        model.rolling_mean_name = 'non_existent_feature'
        with self.assertRaises(ValueError):
            model.predict(self.X)

class TestModelTraining(unittest.TestCase):
    """Tests for the model training functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample data for testing
        np.random.seed(42)
        
        # Create a temporary directory for model output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create sample dataframe
        dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
        countries = ['USA', 'CAN', 'MEX']
        
        data = []
        for country in countries:
            for date in dates:
                # Add some random data with trends and seasonality
                base = 100 if country == 'USA' else (80 if country == 'CAN' else 60)
                trend = 0.5 * (date.month_name() == 'January')
                seasonal = 5 * np.sin(2 * np.pi * date.month / 12)
                noise = np.random.randn() * 2
                
                value = base + trend + seasonal + noise
                
                data.append({
                    'date': date,
                    'country': country,
                    'country_code': country,
                    'value': value,
                    'year': date.year,
                    'month': date.month,
                    'quarter': (date.month - 1) // 3 + 1
                })
        
        self.df = pd.DataFrame(data)
        
        # Save dataframe to a temporary CSV
        self.temp_csv = os.path.join(self.output_dir, "test_data.csv")
        self.df.to_csv(self.temp_csv, index=False)
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_prepare_features(self):
        """Test feature preparation function"""
        # Prepare features
        X, y = prepare_features(self.df)
        
        # Check output shapes
        self.assertEqual(X.shape[0], y.shape[0])
        
        # Check that lag features were created
        for lag in range(1, 13):
            self.assertIn(f'lag_{lag}', X.columns)
        
        # Check that rolling statistics were created
        for window in [3, 6, 12]:
            self.assertIn(f'rolling_mean_{window}', X.columns)
            self.assertIn(f'rolling_std_{window}', X.columns)
    
    @patch('models.training.LinearRegression')
    @patch('models.training.RandomForestRegressor')
    @patch('models.training.joblib.dump')
    def test_train_and_compare_models(self, mock_dump, mock_rf, mock_lr):
        """Test model training and comparison function"""
        # Mock the models to avoid actual training
        mock_lr_instance = MagicMock()
        mock_rf_instance = MagicMock()
        
        mock_lr.return_value = mock_lr_instance
        mock_rf.return_value = mock_rf_instance
        
        # Mock the predict methods
        mock_lr_instance.predict.return_value = np.random.randn(10) + 100
        mock_rf_instance.predict.return_value = np.random.randn(10) + 100
        
        # Call the function with test data
        with patch('models.training.NaiveForecaster') as mock_naive:
            mock_naive_instance = MagicMock()
            mock_naive.return_value = mock_naive_instance
            mock_naive_instance.predict.return_value = np.random.randn(10) + 100
            
            results = train_and_compare_models(self.temp_csv, self.output_dir)
        
        # Check results
        self.assertIsInstance(results, dict)
        self.assertIn('baseline', results)
        self.assertIn('linear_regression', results)
        self.assertIn('random_forest', results)
        
        # Check that metrics were calculated
        for model_name, model_info in results.items():
            self.assertIn('metrics', model_info)
            self.assertIn('rmse', model_info['metrics'])
            self.assertIn('mae', model_info['metrics'])
            self.assertIn('r2', model_info['metrics'])
        
        # Check that models were saved
        self.assertTrue(mock_dump.called)

class TestModelEvaluation(unittest.TestCase):
    """Tests for the model evaluation functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.temp_dir.name, "data/processed")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create sample data
        np.random.seed(42)
        
        # Create sample dataframe
        dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
        countries = ['USA', 'CAN', 'MEX']
        
        data = []
        for country in countries:
            for date in dates:
                # Add some random data with trends and seasonality
                base = 100 if country == 'USA' else (80 if country == 'CAN' else 60)
                trend = 0.5 * (date.month_name() == 'January')
                seasonal = 5 * np.sin(2 * np.pi * date.month / 12)
                noise = np.random.randn() * 2
                
                value = base + trend + seasonal + noise
                
                data.append({
                    'date': date,
                    'country': country,
                    'country_code': country,
                    'value': value,
                    'year': date.year,
                    'month': date.month,
                    'quarter': (date.month - 1) // 3 + 1
                })
        
        self.df = pd.DataFrame(data)
        
        # Save dataframe to a temporary CSV
        self.temp_csv = os.path.join(self.data_dir, "processed_data_20230101_000000.csv")
        self.df.to_csv(self.temp_csv, index=False)
        
        # Create a mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.random.randn(10) + 100
        
        # Save the mock model
        self.model_dir = os.path.join(self.temp_dir.name, "models/model_repository")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "test_model.pkl")
        joblib.dump(self.mock_model, self.model_path)
        
        # Create model info
        self.model_info = {
            "name": "test_model",
            "version": "v1.0",
            "type": "random_forest",
            "metrics": {
                "rmse": 5.0,
                "mae": 4.0,
                "r2": 0.8
            }
        }
        
        # Save model info
        with open(os.path.join(self.model_dir, "test_model_info.json"), 'w') as f:
            import json
            json.dump(self.model_info, f)
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    @patch('models.evaluation.load_model_and_info')
    def test_predict_for_country(self, mock_load):
        """Test prediction for a specific country"""
        # Mock the model loading
        mock_load.return_value = (self.mock_model, self.model_info)
        
        # Call the function with test data
        with patch('models.evaluation.prepare_prediction_features') as mock_prep:
            # Create a mock features dataframe
            mock_features = pd.DataFrame({
                'lag_1': np.random.randn(5) + 100,
                'rolling_mean_3': np.random.randn(5) + 100,
                'month_sin': np.sin(2 * np.pi * np.arange(1, 6) / 12),
                'month_cos': np.cos(2 * np.pi * np.arange(1, 6) / 12),
                'year': [2023] * 5,
                'month': list(range(1, 6)),
                'quarter': [1, 1, 1, 2, 2],
                'date': pd.date_range(start='2023-01-01', periods=5, freq='MS')
            })
            mock_prep.return_value = mock_features
            
            predictions, model_info = predict_for_country('USA', 5, self.mock_model, self.model_path)
        
        # Check results
        self.assertEqual(len(predictions), 5)
        for pred in predictions:
            self.assertIn('date', pred)
            self.assertIn('value', pred)
            self.assertIn('year', pred)
            self.assertIn('month', pred)
            self.assertIn('quarter', pred)
        
        self.assertEqual(model_info, self.model_info)

if __name__ == "__main__":
    unittest.main()
