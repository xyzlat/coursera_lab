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
