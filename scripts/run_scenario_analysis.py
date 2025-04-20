"""Model Diagnostics for Coal Price Forecasting."""
import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils.logging_config import setup_logging, load_config, CONFIG_PATH
from src.modeling.ts_models import TSModels
from src.modeling.model_diagnostics import (
    plot_model_diagnostics,
    compare_in_out_sample_metrics,
    save_model_summary
)

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Run the model diagnostics pipeline."""
    try:
        # Load configuration
        config = load_config(CONFIG_PATH)
        if config is None:
            raise ValueError("Failed to load configuration")
            
        # Load data
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
        
        # Load and fit the model
        model = TSModels(config)
        target_var = config['variables']['target_variable']
        X_train = train_data.drop(columns=[target_var])
        y_train = train_data[target_var]
        
        logger.info("Fitting ARIMAX model...")
        model.fit(X_train, y_train)
        
        # Save detailed model summary
        save_model_summary(
            model,
            os.path.join(PROJECT_ROOT, 'reports/arimax_model_summary.txt')
        )
        
        # Generate and save model diagnostics
        train_pred, _ = model.predict(X_train)
        residuals = y_train.values - train_pred.values
        
        # Fix: calling plot_model_diagnostics with the correct number of arguments
        diagnostic_results = plot_model_diagnostics(residuals)
        
        # Handle diagnostic results manually
        shapiro_test = stats.shapiro(residuals)
        ljung_box_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
        
        logger.info("Model Diagnostic Test Results:")
        logger.info(f"Shapiro-Wilk test p-value: {shapiro_test[1]:.4f}")
        for i, lag in enumerate([10, 20, 30]):
            pval = ljung_box_test['lb_pvalue'].iloc[i]
            logger.info(f"Ljung-Box test ({lag}): {pval:.4f}")
            
        # Compare in-sample vs out-of-sample performance
        metric_comparison = compare_in_out_sample_metrics(
            model,
            train_data,
            test_data,
            target_var
        )
        
        logger.info("\nPerformance Metric Comparison:")
        logger.info("Training Set Metrics:")
        for metric, value in metric_comparison['train'].items():
            logger.info(f"{metric}: {value:.4f}")
            
        logger.info("\nTest Set Metrics:")
        for metric, value in metric_comparison['test'].items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info("Model diagnostics completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model diagnostics: {e}")
        raise

if __name__ == "__main__":
    main()