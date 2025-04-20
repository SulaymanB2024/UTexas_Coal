"""Evaluate enhanced SARIMAX model and compare with previous ARIMAX results."""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')

from src.utils.logging_config import setup_logging, load_config
from src.modeling.sarimax_model import EnhancedSARIMAX
from src.modeling.model_diagnostics import (
    plot_model_diagnostics,
    compare_in_out_sample_metrics
)
from src.data_processing.data_processor import DataProcessor

logger = logging.getLogger(__name__)

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets."""
    train_data = pd.read_csv(
        os.path.join(PROJECT_ROOT, 'data/processed/processed_train.csv')
    )
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)
    
    test_data = pd.read_csv(
        os.path.join(PROJECT_ROOT, 'data/processed/processed_test.csv')
    )
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    test_data.set_index('Date', inplace=True)
    
    return train_data, test_data

def compare_model_metrics(
    old_metrics: Dict,
    new_metrics: Dict,
    output_file: str
) -> None:
    """Compare and save metrics from old and new models."""
    comparison = pd.DataFrame({
        'ARIMAX_Train': old_metrics['train'],
        'ARIMAX_Test': old_metrics['test'],
        'SARIMAX_Train': new_metrics['train'],
        'SARIMAX_Test': new_metrics['test']
    }).round(4)
    
    comparison.to_csv(output_file)
    logger.info(f"Saved model comparison to {output_file}")
    
    # Log comparison
    logger.info("\nModel Comparison:")
    for metric in comparison.index:
        logger.info(f"\n{metric}:")
        logger.info(f"ARIMAX - Train: {comparison.loc[metric, 'ARIMAX_Train']:.4f}, "
                   f"Test: {comparison.loc[metric, 'ARIMAX_Test']:.4f}")
        logger.info(f"SARIMAX - Train: {comparison.loc[metric, 'SARIMAX_Train']:.4f}, "
                   f"Test: {comparison.loc[metric, 'SARIMAX_Test']:.4f}")
        improvement = ((comparison.loc[metric, 'SARIMAX_Test'] - 
                       comparison.loc[metric, 'ARIMAX_Test']) / 
                      comparison.loc[metric, 'ARIMAX_Test'] * 100)
        logger.info(f"Test Set Improvement: {improvement:.2f}%")

def log_diagnostic_results(diagnostics: Dict) -> None:
    """Log results of diagnostic tests."""
    logger.info("\nModel Diagnostic Test Results:")
    
    if 'shapiro_wilk' in diagnostics:
        logger.info(f"Shapiro-Wilk test p-value: {diagnostics['shapiro_wilk']['pvalue']:.4f}")
        
    if 'ljung_box' in diagnostics:
        for i, lag in enumerate(diagnostics['ljung_box']['lags']):
            logger.info(f"Ljung-Box test (lag_{lag}): {diagnostics['ljung_box']['pvalue'][i]:.4f}")

def main():
    """Main execution function."""
    # Load config and setup logging
    config = load_config(CONFIG_PATH)
    setup_logging(config)
    
    try:
        # Load data using existing function
        train_data, test_data = load_data()
        
        target_var = config['variables']['target_variable']
        
        # Initialize and fit model
        model = EnhancedSARIMAX(config)
        logger.info("Fitting Enhanced SARIMAX model...")
        
        # Separate features and target
        X_train = train_data.drop(columns=[target_var])
        y_train = train_data[target_var]
        X_test = test_data.drop(columns=[target_var])
        y_test = test_data[target_var]
        
        # Fit model
        model.fit(X_train, y_train, freq='ME')
        
        # Get diagnostics
        diagnostics = model.get_diagnostics()
        log_diagnostic_results(diagnostics)
        
        # Plot diagnostics
        plot_model_diagnostics(
            residuals=diagnostics['residuals'],
            config=config
        )
        
        # Generate and evaluate predictions
        train_pred, _ = model.predict(X_train)
        test_pred, _ = model.predict(X_test)
        
        # Ensure index alignment
        train_pred.index = X_train.index
        test_pred.index = X_test.index
        
        # Calculate metrics
        metrics = compare_in_out_sample_metrics(
            model=model,
            train_data=train_data,
            test_data=test_data,
            target_var=target_var
        )
        
        logger.info("\nModel Performance Metrics:")
        for split, split_metrics in metrics.items():
            logger.info(f"\n{split.title()} Set Metrics:")
            for metric, value in split_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
                
    except Exception as e:
        logger.error(f"Error in SARIMAX evaluation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()