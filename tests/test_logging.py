import os
import sys
import unittest
import tempfile
import logging
from unittest.mock import patch, MagicMock
import io

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from monitoring.logger import (
    setup_logger, 
    create_directory_if_not_exists, 
    ProductionLogger, 
    TestLogger,
    get_test_logger,
    get_production_logger
)

class TestLogging(unittest.TestCase):
    """Tests for the logging module"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for logs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = self.temp_dir.name
        
        # Test log file path
        self.log_file = os.path.join(self.log_dir, "test.log")
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_create_directory(self):
        """Test directory creation function"""
        # Test directory path
        test_dir = os.path.join(self.temp_dir.name, "test_subdir")
        
        # Directory shouldn't exist initially
        self.assertFalse(os.path.exists(test_dir))
        
        # Create directory
        create_directory_if_not_exists(test_dir)
        
        # Check that directory was created
        self.assertTrue(os.path.exists(test_dir))
        
        # Call function again (should not raise errors)
        create_directory_if_not_exists(test_dir)
    
    def test_setup_logger(self):
        """Test logger setup function"""
        # Create logger
        logger = setup_logger("test_logger", self.log_file)
        
        # Check logger properties
        self.assertEqual(logger.name, "test_logger")
        self.assertEqual(logger.level, logging.INFO)
        
        # Check that handlers were added
        self.assertEqual(len(logger.handlers), 2)  # File handler and console handler
        
        # Log a message
        test_message = "Test log message"
        logger.info(test_message)
        
        # Check that message was written to file
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn(test_message, log_content)
    
    def test_production_logger(self):
        """Test the production logger class"""
        # Create production logger
        prod_logger = ProductionLogger("prod_test", self.log_file, "test_env")
        
        # Check logger properties
        self.assertEqual(prod_logger.environment, "test_env")
        
        # Log messages with metadata
        prod_logger.info("Test info", {"user": "tester"})
        prod_logger.warning("Test warning", {"severity": "medium"})
        prod_logger.error("Test error", {"code": "E123"})
        
        # Check that messages were written to file
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("Test info", log_content)
            self.assertIn("user=tester", log_content)
            self.assertIn("environment=test_env", log_content)
            self.assertIn("Test warning", log_content)
            self.assertIn("severity=medium", log_content)
            self.assertIn("Test error", log_content)
            self.assertIn("code=E123", log_content)
    
    def test_test_logger(self):
        """Test the test logger class"""
        # Capture stdout to verify console logging
        captured_output = io.StringIO()
        with patch('sys.stdout', new=captured_output):
            # Create test logger
            test_logger = TestLogger("test_logger")
            
            # Log a message
            test_message = "Test log message for test environment"
            test_logger.info(test_message)
            
            # Check output
            output = captured_output.getvalue()
            self.assertIn(test_message, output)
            self.assertIn("TEST", output)
    
    def test_logger_factory_functions(self):
        """Test the logger factory functions"""
        # Test the get_test_logger function
        test_logger = get_test_logger("factory_test")
        self.assertIsInstance(test_logger, TestLogger)
        self.assertEqual(test_logger.logger.name, "test_factory_test")
        
        # Test the get_production_logger function
        prod_logger = get_production_logger("factory_prod", self.log_file)
        self.assertIsInstance(prod_logger, ProductionLogger)
        self.assertEqual(prod_logger.environment, "production")
        
        # Log a message with the production logger
        prod_logger.info("Factory logger test")
        
        # Check that the message was written to file
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("Factory logger test", log_content)

if __name__ == "__main__":
    unittest.main()
