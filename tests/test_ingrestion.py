import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from data.ingestion import ingest_data, validate_data, create_directory_if_not_exists

class TestDataIngestion(unittest.TestCase):
    """Tests for the data ingestion module"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = os.path.join(self.temp_dir.name, "processed")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=12, freq='MS'),
            'country': ['USA'] * 12,
            'value': np.random.rand(12) * 100
        })
        
        # Create sample CSV file
        self.csv_path = os.path.join(self.temp_dir.name, "test_data.csv")
        self.sample_data.to_csv(self.csv_path, index=False)
        
        # Create Excel file
        self.excel_path = os.path.join(self.temp_dir.name, "test_data.xlsx")
        self.sample_data.to_excel(self.excel_path, index=False)
        
        # Create JSON file
        self.json_path = os.path.join(self.temp_dir.name, "test_data.json")
        self.sample_data.to_json(self.json_path, orient='records')
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_validate_data_valid(self):
        """Test data validation with valid data"""
        # Create valid dataframe
        valid_df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=5, freq='MS'),
            'country': ['USA'] * 5,
            'value': [100, 105, 110, 115, 120]
        })
        
        # Validate the data
        is_valid, error_message = validate_data(valid_df)
        
        # Check results
        self.assertTrue(is_valid)
        self.assertIsNone(error_message)
    
    def test_validate_data_invalid(self):
        """Test data validation with invalid data"""
        # Create invalid dataframe (missing required column)
        invalid_df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=5, freq='MS'),
            'country': ['USA'] * 5
            # Missing 'value' column
        })
        
        # Validate the data
        is_valid, error_message = validate_data(invalid_df)
        
        # Check results
        self.assertFalse(is_valid)
        self.assertIn("Missing required columns", error_message)
        
        # Create invalid dataframe (missing values)
        invalid_df2 = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=5, freq='MS'),
            'country': ['USA', 'USA', None, 'USA', 'USA'],
            'value': [100, 105, 110, 115, 120]
        })
        
        # Validate the data
        is_valid, error_message = validate_data(invalid_df2)
        
        # Check results
        self.assertFalse(is_valid)
        self.assertIn("contains missing values", error_message)
        
        # Create invalid dataframe (invalid date format)
        invalid_df3 = pd.DataFrame({
            'date': ['2020-01-01', '2020-02-01', 'invalid', '2020-04-01', '2020-05-01'],
            'country': ['USA'] * 5,
            'value': [100, 105, 110, 115, 120]
        })
        
        # Validate the data
        is_valid, error_message = validate_data(invalid_df3)
        
        # Check results
        self.assertFalse(is_valid)
        self.assertIn("Date column has invalid format", error_message)
    
    def test_ingest_data_csv(self):
        """Test data ingestion with CSV file"""
        # Ingest the data
        output_path = ingest_data(self.csv_path, self.output_dir)
        
        # Check that output file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check that countries file was created
        countries_path = os.path.join(self.output_dir, "countries.csv")
        self.assertTrue(os.path.exists(countries_path))
        
        # Load the processed data
        processed_data = pd.read_csv(output_path)
        
        # Check that required columns are present
        self.assertIn('date', processed_data.columns)
        self.assertIn('country', processed_data.columns)
        self.assertIn('value', processed_data.columns)
        self.assertIn('year', processed_data.columns)
        self.assertIn('month', processed_data.columns)
        self.assertIn('quarter', processed_data.columns)
        self.assertIn('country_code', processed_data.columns)
        
        # Check row count
        self.assertEqual(len(processed_data), len(self.sample_data))
    
    def test_ingest_data_excel(self):
        """Test data ingestion with Excel file"""
        # Ingest the data
        output_path = ingest_data(self.excel_path, self.output_dir)
        
        # Check that output file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Load the processed data
        processed_data = pd.read_csv(output_path)
        
        # Check that required columns are present
        self.assertIn('date', processed_data.columns)
        self.assertIn('country', processed_data.columns)
        self.assertIn('value', processed_data.columns)
        
        # Check row count
        self.assertEqual(len(processed_data), len(self.sample_data))
    
    def test_ingest_data_json(self):
        """Test data ingestion with JSON file"""
        # Ingest the data
        output_path = ingest_data(self.json_path, self.output_dir)
        
        # Check that output file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Load the processed data
        processed_data = pd.read_csv(output_path)
        
        # Check that required columns are present
        self.assertIn('date', processed_data.columns)
        self.assertIn('country', processed_data.columns)
        self.assertIn('value', processed_data.columns)
        
        # Check row count
        self.assertEqual(len(processed_data), len(self.sample_data))
    
    def test_ingest_data_unsupported_format(self):
        """Test data ingestion with unsupported file format"""
        # Create an unsupported file
        unsupported_path = os.path.join(self.temp_dir.name, "test_data.txt")
        with open(unsupported_path, 'w') as f:
            f.write("This is a test file")
        
        # Attempt to ingest the data (should raise ValueError)
        with self.assertRaises(ValueError):
            ingest_data(unsupported_path, self.output_dir)
    
    def test_ingest_data_invalid_data(self):
        """Test data ingestion with invalid data"""
        # Create invalid CSV file (missing required column)
        invalid_path = os.path.join(self.temp_dir.name, "invalid_data.csv")
        invalid_df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=5, freq='MS'),
            'country': ['USA'] * 5
            # Missing 'value' column
        })
        invalid_df.to_csv(invalid_path, index=False)
        
        # Attempt to ingest the data (should raise ValueError)
        with self.assertRaises(ValueError):
            ingest_data(invalid_path, self.output_dir)

if __name__ == "__main__":
    unittest.main()
