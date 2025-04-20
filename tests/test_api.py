import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import pandas as pd
from fastapi.testclient import TestClient

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FastAPI app
from api.app import app

# Create test client
client = TestClient(app)

class TestAPI(unittest.TestCase):
    """Tests for the API endpoints"""
    
    @patch('api.app.predict_for_country')
    def test_predict_for_country(self, mock_predict_for_country):
        """Test the predict endpoint for a specific country"""
        # Mock the predict_for_country function
        mock_predictions = [
            {"date": "2023-01-01", "value": 100.5, "year": 2023, "month": 1, "quarter": 1},
            {"date": "2023-02-01", "value": 105.2, "year": 2023, "month": 2, "quarter": 1}
        ]
        mock_model_info = {"version": "test_model_v1", "type": "random_forest"}
        
        mock_predict_for_country.return_value = (mock_predictions, mock_model_info)
        
        # Make request to the API
        response = client.get("/predict/USA?horizon=2")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure
        self.assertEqual(data["country"], "USA")
        self.assertEqual(len(data["predictions"]), 2)
        self.assertEqual(data["model_version"], "test_model_v1")
        self.assertEqual(data["model_type"], "random_forest")
        
        # Verify mock was called correctly
        mock_predict_for_country.assert_called_once()
        args, kwargs = mock_predict_for_country.call_args
        self.assertEqual(args[0], "USA")  # Country
        self.assertEqual(args[1], 2)      # Horizon
    
    @patch('api.app.predict_for_country')
    def test_predict_for_country_error(self, mock_predict_for_country):
        """Test predict endpoint error handling"""
        # Mock the function to raise an error
        mock_predict_for_country.side_effect = ValueError("Invalid country code")
        
        # Make request to the API
        response = client.get("/predict/XYZ?horizon=12")
        
        # Check response
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["detail"], "Invalid country code")
    
    @patch('api.app.predict_for_country')
    @patch('api.app.pd.read_csv')
    def test_predict_all(self, mock_read_csv, mock_predict_for_country):
        """Test the predict all endpoint"""
        # Mock the dataframe with countries
        mock_countries = pd.DataFrame({
            "country_code": ["USA", "CAN", "MEX"],
            "country": ["United States", "Canada", "Mexico"]
        })
        mock_read_csv.return_value = mock_countries
        
        # Mock the predict_for_country function
        mock_predictions = [
            {"date": "2023-01-01", "value": 100.5, "year": 2023, "month": 1, "quarter": 1}
        ]
        mock_model_info = {"version": "test_model_v1", "type": "random_forest"}
        
        mock_predict_for_country.return_value = (mock_predictions, mock_model_info)
        
        # Make request to the API
        response = client.get("/predict/?horizon=1")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure
        self.assertEqual(len(data), 3)  # 3 countries
        
        for country_pred in data:
            self.assertIn(country_pred["country"], ["USA", "CAN", "MEX"])
            self.assertEqual(len(country_pred["predictions"]), 1)
            self.assertEqual(country_pred["model_version"], "test_model_v1")
            self.assertEqual(country_pred["model_type"], "random_forest")
        
        # Verify mock was called correctly - 3 times for 3 countries
        self.assertEqual(mock_predict_for_country.call_count, 3)
    
    def test_health_check(self):
        """Test the health check endpoint"""
        # This assumes a model is loaded in the app.py, which might not be true in tests
        # Mock the app.model to avoid errors
        with patch('api.app.model', "mock_model"):
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["model_loaded"])
    
    def test_health_check_no_model(self):
        """Test the health check endpoint when no model is loaded"""
        # Mock the app.model to be None
        with patch('api.app.model', None):
            response = client.get("/health")
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertEqual(data["detail"], "Model not loaded")

if __name__ == "__main__":
    unittest.main()
