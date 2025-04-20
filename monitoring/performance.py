import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from models.evaluation import evaluate_model_performance
from monitoring.logger import setup_logger

# Set up logger
logger = setup_logger("performance_monitoring", "logs/monitoring.log")

def create_directory_if_not_exists(directory_path: str) -> None:
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def log_performance_metrics(metrics: Dict[str, Any], 
                           output_dir: str = "monitoring/performance_logs") -> str:
    """
    Log model performance metrics to a file
    
    Args:
        metrics: Performance metrics to log
        output_dir: Directory to save the performance logs
        
    Returns:
        Path to the saved performance log file
    """
    # Create output directory if it doesn't exist
    create_directory_if_not_exists(output_dir)
    
    # Add timestamp to the metrics
    metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_metrics_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save metrics to file
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Performance metrics logged to {filepath}")
    return filepath

def capture_performance_snapshot(model_path: str = "models/model_repository/best_model.pkl", 
                                 data_path: str = "data/processed",
                                 output_dir: str = "monitoring/performance_logs") -> Dict[str, Any]:
    """
    Capture a snapshot of the current model performance
    
    Args:
        model_path: Path to the model to evaluate
        data_path: Path to the processed data directory
        output_dir: Directory to save the performance logs
        
    Returns:
        Dictionary with performance metrics and metadata
    """
    logger.info(f"Capturing performance snapshot for model: {model_path}")
    
    try:
        # Measure execution time
        start_time = time.time()
        
        # Evaluate model performance
        performance_metrics = evaluate_model_performance(model_path, data_path)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Get model info
        with open(model_path.replace(".pkl", "_info.json"), 'r') as f:
            model_info = json.load(f)
        
        # Create snapshot
        snapshot = {
            'model_name': model_info.get('name', 'unknown'),
            'model_version': model_info.get('version', 'unknown'),
            'model_type': model_info.get('type', 'unknown'),
            'execution_time_seconds': execution_time,
            'data_path': data_path,
            'metrics': performance_metrics
        }
        
        # Log the snapshot
        log_filepath = log_performance_metrics(snapshot, output_dir)
        
        logger.info(f"Performance snapshot completed in {execution_time:.2f} seconds")
        return snapshot
        
    except Exception as e:
        logger.error(f"Error capturing performance snapshot: {str(e)}")
        raise

def create_performance_dashboard(metrics_dir: str = "monitoring/performance_logs",
                                output_dir: str = "monitoring/reports") -> str:
    """
    Create a visual performance dashboard from logged metrics
    
    Args:
        metrics_dir: Directory containing performance metrics logs
        output_dir: Directory to save the generated reports
        
    Returns:
        Path to the generated dashboard file
    """
    logger.info("Creating performance dashboard")
    
    try:
        # Create output directory if it doesn't exist
        create_directory_if_not_exists(output_dir)
        
        # Find all metrics files
        metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.json')]
        if not metrics_files:
            logger.warning("No performance metrics files found")
            return None
        
        # Sort files by timestamp (assuming filename format includes timestamp)
        metrics_files.sort()
        
        # Load all metrics into a list
        all_metrics = []
        for file in metrics_files:
            with open(os.path.join(metrics_dir, file), 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
        
        # Create a DataFrame for overall metrics
        overall_df = pd.DataFrame([
            {
                'timestamp': m['timestamp'],
                'model_name': m['model_name'],
                'model_version': m['model_version'],
                'rmse': m['metrics']['overall']['rmse'],
                'mae': m['metrics']['overall']['mae'],
                'r2': m['metrics']['overall']['r2'],
                'execution_time': m['execution_time_seconds']
            }
            for m in all_metrics
        ])
        
        # Create a DataFrame for country-specific metrics
        country_metrics = []
        for m in all_metrics:
            timestamp = m['timestamp']
            model_name = m['model_name']
            for country, metrics in m['metrics'].get('by_country', {}).items():
                country_metrics.append({
                    'timestamp': timestamp,
                    'model_name': model_name,
                    'country': country,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'sample_size': metrics.get('sample_size', 0)
                })
        
        country_df = pd.DataFrame(country_metrics)
        
        # Generate plots
        plt.figure(figsize=(12, 10))
        
        # Plot overall metrics over time
        plt.subplot(2, 2, 1)
        sns.lineplot(data=overall_df, x='timestamp', y='rmse', marker='o')
        plt.title('Overall RMSE Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.subplot(2, 2, 2)
        sns.lineplot(data=overall_df, x='timestamp', y='r2', marker='o')
        plt.title('Overall R² Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Plot metrics by country (most recent snapshot)
        if not country_df.empty:
            recent_country_metrics = country_df[country_df['timestamp'] == country_df['timestamp'].max()]
            
            plt.subplot(2, 2, 3)
            country_plot = sns.barplot(data=recent_country_metrics, x='country', y='rmse')
            plt.title('RMSE by Country (Latest)')
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            plt.subplot(2, 2, 4)
            execution_time_plot = sns.lineplot(data=overall_df, x='timestamp', y='execution_time', marker='o')
            plt.title('Model Execution Time')
            plt.xticks(rotation=45)
            plt.ylabel('Seconds')
            plt.tight_layout()
        
        # Save the dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_path = os.path.join(output_dir, f"performance_dashboard_{timestamp}.png")
        plt.savefig(dashboard_path)
        plt.close()
        
        # Also generate CSV reports
        overall_csv_path = os.path.join(output_dir, f"overall_metrics_{timestamp}.csv")
        country_csv_path = os.path.join(output_dir, f"country_metrics_{timestamp}.csv")
        
        overall_df.to_csv(overall_csv_path, index=False)
        if not country_df.empty:
            country_df.to_csv(country_csv_path, index=False)
        
        logger.info(f"Performance dashboard created: {dashboard_path}")
        return dashboard_path
        
    except Exception as e:
        logger.error(f"Error creating performance dashboard: {str(e)}")
        raise

def monitor_model_drift(threshold: float = 0.1,
                       metrics_dir: str = "monitoring/performance_logs") -> Dict[str, Any]:
    """
    Monitor for model drift by comparing recent performance with historical performance
    
    Args:
        threshold: Threshold for significant drift detection
        metrics_dir: Directory containing performance metrics logs
        
    Returns:
        Dictionary with drift detection results
    """
    logger.info(f"Monitoring model drift with threshold: {threshold}")
    
    try:
        # Find all metrics files
        metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.json')]
        if len(metrics_files) < 2:
            logger.warning("Not enough data to detect drift (need at least 2 snapshots)")
            return {'drift_detected': False, 'reason': 'insufficient_data'}
        
        # Sort files by timestamp
        metrics_files.sort()
        
        # Load the most recent and previous metrics
        with open(os.path.join(metrics_dir, metrics_files[-1]), 'r') as f:
            current_metrics = json.load(f)
        
        with open(os.path.join(metrics_dir, metrics_files[-2]), 'r') as f:
            previous_metrics = json.load(f)
        
        # Compare overall metrics
        current_rmse = current_metrics['metrics']['overall']['rmse']
        previous_rmse = previous_metrics['metrics']['overall']['rmse']
        
        rmse_change = (current_rmse - previous_rmse) / previous_rmse
        
        drift_detected = abs(rmse_change) > threshold
        drift_direction = 'worse' if rmse_change > 0 else 'better'
        
        # Create drift report
        drift_report = {
            'drift_detected': drift_detected,
            'threshold': threshold,
            'current_timestamp': current_metrics['timestamp'],
            'previous_timestamp': previous_metrics['timestamp'],
            'current_rmse': current_rmse,
            'previous_rmse': previous_rmse,
            'rmse_change': rmse_change,
            'change_percent': rmse_change * 100,
            'direction': drift_direction if drift_detected else 'stable'
        }
        
        # Log drift information
        if drift_detected:
            logger.warning(f"Model drift detected! RMSE changed by {rmse_change*100:.2f}% ({drift_direction})")
        else:
            logger.info(f"No significant model drift detected (RMSE changed by {rmse_change*100:.2f}%)")
        
        return drift_report
        
    except Exception as e:
        logger.error(f"Error monitoring model drift: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model performance monitoring")
    parser.add_argument("--model", default="models/model_repository/best_model.pkl", help="Path to the model file")
    parser.add_argument("--data", default="data/processed", help="Path to processed data directory")
    parser.add_argument("--output", default="monitoring/performance_logs", help="Output directory for performance logs")
    parser.add_argument("--threshold", type=float, default=0.1, help="Drift detection threshold")
    parser.add_argument("--dashboard", action="store_true", help="Generate performance dashboard")
    
    args = parser.parse_args()
    
    try:
        # Capture performance snapshot
        snapshot = capture_performance_snapshot(args.model, args.data, args.output)
        print(f"Performance snapshot completed: Overall RMSE = {snapshot['metrics']['overall']['rmse']:.4f}")
        
        # Check for model drift
        drift_report = monitor_model_drift(args.threshold, args.output)
        if drift_report['drift_detected']:
            print(f"ALERT: Model drift detected! RMSE changed by {drift_report['change_percent']:.2f}% ({drift_report['direction']})")
        else:
            print("No significant model drift detected")
        
        # Generate performance dashboard if requested
        if args.dashboard:
            dashboard_path = create_performance_dashboard(args.output, "monitoring/reports")
            print(f"Performance dashboard created: {dashboard_path}")
            
    except Exception as e:
        print(f"Error in performance monitoring: {str(e)}")
        exit(1)
