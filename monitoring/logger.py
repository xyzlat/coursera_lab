import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def create_directory_if_not_exists(directory_path: str) -> None:
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def setup_logger(name: str, log_file: str, 
                level=logging.INFO, 
                max_bytes=10485760,  # 10MB
                backup_count=5) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Name of the logger
        log_file: Path to the log file
        level: Logging level
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    create_directory_if_not_exists(log_dir)
    
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create handlers
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatters and add them to handlers
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(file_format)
    console_handler.setFormatter(console_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class ProductionLogger:
    """
    A logger class for production environments with additional metadata
    """
    
    def __init__(self, name: str, log_file: str, environment: str = "production"):
        self.logger = setup_logger(name, log_file)
        self.environment = environment
        
    def _format_message(self, message: str, metadata: dict = None) -> str:
        """Format message with metadata"""
        if metadata is None:
            metadata = {}
        
        # Add default metadata
        metadata.update({
            'environment': self.environment,
            'timestamp': datetime.now().isoformat()
        })
        
        # Format metadata as string
        metadata_str = ', '.join([f'{k}={v}' for k, v in metadata.items()])
        
        return f"{message} [{metadata_str}]"
    
    def info(self, message: str, metadata: dict = None):
        """Log info message with metadata"""
        self.logger.info(self._format_message(message, metadata))
    
    def warning(self, message: str, metadata: dict = None):
        """Log warning message with metadata"""
        self.logger.warning(self._format_message(message, metadata))
    
    def error(self, message: str, metadata: dict = None):
        """Log error message with metadata"""
        self.logger.error(self._format_message(message, metadata))
    
    def debug(self, message: str, metadata: dict = None):
        """Log debug message with metadata"""
        self.logger.debug(self._format_message(message, metadata))

class TestLogger:
    """
    A logger class for test environments that doesn't write to production logs
    """
    
    def __init__(self, name: str):
        # Create a console-only logger for tests
        self.logger = logging.getLogger(f"test_{name}")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers = []
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - TEST - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)

def get_test_logger(name: str) -> TestLogger:
    """Get a logger instance for testing"""
    return TestLogger(name)

def get_production_logger(name: str, log_file: str) -> ProductionLogger:
    """Get a logger instance for production"""
    return ProductionLogger(name, log_file)
