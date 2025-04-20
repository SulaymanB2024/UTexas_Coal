"""Evaluate the economic value of SARIMAX forecasts using a directional trading strategy."""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')

from src.utils.logging_config import setup_logging, load_config
from src.modeling.sarimax_model import EnhancedSARIMAX
from src.modeling.trading_strategy import TradingStrategy
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

def main():
    """Main execution function."""
    # Load config and setup logging
    config = load_config(CONFIG_PATH)
    setup_logging(config)
    
    try:
        # Load data
        train_data, test_data = load_data()
        
        target_var = config['variables']['target_variable']
        
        # Separate features and target
        X_train = train_data.drop(columns=[target_var])
        y_train = train_data[target_var]
        X_test = test_data.drop(columns=[target_var])
        y_test = test_data[target_var]
        
        # Initialize model with standard SARIMAX(1,1,1)(1,0,1,12) parameters
        # This is the accepted model based on information criteria per requirements
        logger.info("Loading SARIMAX(1,1,1)(1,0,1,12) model...")
        fixed_model = EnhancedSARIMAX(
            config,
            seasonal=True,
            m=12
        )
        
        # Manually set the model parameters instead of grid searching
        fixed_model.best_order = (1, 1, 1)  # p, d, q
        fixed_model.best_seasonal_order = (1, 0, 1, 12)  # P, D, Q, m
        
        # Fit model
        logger.info("Fitting model with fixed parameters...")
        fixed_model.fit(X_train, y_train, freq='ME')
        
        # Get diagnostics and residuals for threshold calibration
        diagnostics = fixed_model.get_diagnostics()
        train_residuals = diagnostics['residuals']
        
        # Generate in-sample and out-of-sample predictions
        logger.info("Generating predictions for trading strategy evaluation...")
        train_pred, _ = fixed_model.predict(X_train)
        test_pred, _ = fixed_model.predict(X_test)
        
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(PROJECT_ROOT, 'reports')
        figures_dir = os.path.join(reports_dir, 'figures')
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        # Define strategies to evaluate
        strategies = [
            {
                "name": "directional",
                "params": {
                    "strategy_type": "directional",
                    "transaction_cost": 0.001,
                    "slippage": 0.0005,
                    "position_size": 1,
                    "risk_free_rate": 0.02,
                    "trading_frequency": 'ME'
                },
                "output_prefix": "economic_value_directional"
            },
            {
                "name": "threshold",
                "params": {
                    "strategy_type": "threshold",
                    "transaction_cost": 0.001,
                    "slippage": 0.0005,
                    "position_size": 1,
                    "threshold_sd_multiple": 0.5,
                    "risk_free_rate": 0.02,
                    "trading_frequency": 'ME'
                },
                "output_prefix": "economic_value_threshold"
            },
            {
                "name": "combined",
                "params": {
                    "strategy_type": "both",
                    "transaction_cost": 0.001,
                    "slippage": 0.0005,
                    "position_size": 1,
                    "threshold_sd_multiple": 0.5,
                    "risk_free_rate": 0.02,
                    "trading_frequency": 'ME'
                },
                "output_prefix": "economic_value_combined"
            }
        ]
        
        # Evaluate each strategy
        for strategy_config in strategies:
            strategy_name = strategy_config["name"]
            params = strategy_config["params"] 
            output_prefix = strategy_config["output_prefix"]
            
            logger.info(f"\n========== EVALUATING {strategy_name.upper()} STRATEGY ==========")
            
            # Initialize trading strategy with parameters
            strategy = TradingStrategy(
                config=config,
                **params
            )
            
            # Evaluate strategy on test data
            logger.info(f"Evaluating {strategy_name} trading strategy on test data...")
            results = strategy.evaluate_strategy(
                train_residuals=train_residuals,
                actual_prices=y_test,
                forecast_prices=test_pred
            )
            
            # Plot results
            logger.info(f"Generating performance plots for {strategy_name} strategy...")
            strategy.plot_performance(
                results['trading_df'],
                output_path=os.path.join(figures_dir, f'{output_prefix}_performance.html')
            )
            
            # Save results
            logger.info(f"Saving {strategy_name} strategy evaluation results...")
            output_file = os.path.join(reports_dir, f'{output_prefix}_results.csv')
            strategy.save_results(
                results['trading_df'],
                output_file=output_file
            )
            
            logger.info(f"{strategy_name.capitalize()} strategy analysis completed. Results saved to {output_file}")
            logger.info("=" * 60)
        
        logger.info(f"All economic value analyses completed.")
        
    except Exception as e:
        logger.error(f"Error in economic value evaluation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()