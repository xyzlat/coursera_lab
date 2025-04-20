import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
from typing import Tuple, Optional

from monitoring.logger import setup_logger

# Set up logger
logger = setup_logger("data_ingestion", "logs/ingestion.log")

def create_directory_if_not_exists(directory_path: str) -> None:
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def validate_data(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate the ingested data for quality issues
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for empty dataframe
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check for required columns
    required_columns = ['date', 'country', 'value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check for missing values in key columns
    for col in required_columns:
        if df[col].isna().sum() > 0:
            return False, f"Column '{col}' contains missing values"
    
    # Check date format
    try:
        pd.to_datetime(df['date'])
    except:
        return False, "Date column has invalid format"
    
    return True, None

def ingest_data(source_path: str, output_dir: str = "data/processed") -> str:
    """
    Ingest data from source, validate, transform, and save
    
    Args:
        source_path: Path to the source data
        output_dir: Directory to save processed data
    
    Returns:
        Path to the processed data file
    """
    logger.info(f"Starting data ingestion from {source_path}")
    
    # Create output directory if it doesn't exist
    create_directory_if_not_exists(output_dir)
    
    try:
        # Determine file type and read accordingly
        file_extension = os.path.splitext(source_path)[1].lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(source_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(source_path)
        elif file_extension == '.json':
            df = pd.read_json(source_path)
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Validate the data
        is_valid, error_message = validate_data(df)
        if not is_valid:
            logger.error(f"Data validation failed: {error_message}")
            raise ValueError(f"Data validation failed: {error_message}")
        
        # Process the data
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract additional features from date
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Create country code column (assuming country is full name)
        df['country_code'] = df['country'].str.upper().str[:3]
        
        # Log data shape
        logger.info(f"Processed data shape: {df.shape}")
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"processed_data_{timestamp}.csv")
        df.to_csv(output_path, index=False)
        
        # Also save a countries reference file for API use
        countries_df = df[['country', 'country_code']].drop_duplicates()
        countries_path = os.path.join(output_dir, "countries.csv")
        countries_df.to_csv(countries_path, index=False)
        
        logger.info(f"Data successfully processed and saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error during data ingestion: {str(e)}")
        raise
        
def main():
    parser = argparse.ArgumentParser(description="Data ingestion script")
    parser.add_argument("--source", required=True, help="Path to source data file")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for processed data")
    
    args = parser.parse_args()
    
    try:
        output_path = ingest_data(args.source, args.output_dir)
        print(f"Data ingestion completed successfully. Processed data saved to: {output_path}")
    except Exception as e:
        print(f"Data ingestion failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
