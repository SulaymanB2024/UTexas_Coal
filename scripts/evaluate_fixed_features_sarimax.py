"""Evaluate enhanced SARIMAX model with fixed features and orders."""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')

from src.utils.logging_config import setup_logging, load_config
from src.modeling.sarimax_model import EnhancedSARIMAX
from src.modeling.model_diagnostics import (
    plot_model_diagnostics,
    compare_in_out_sample_metrics,
    save_model_summary
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

def log_diagnostic_results(diagnostics: Dict) -> None:
    """Log results of diagnostic tests.
    
    Args:
        diagnostics: Dictionary containing diagnostic test results
    """
    logger.info("\nModel Diagnostic Test Results:")
    
    if 'residuals' in diagnostics:
        resid_stats = pd.Series(diagnostics['residuals']).describe()
        logger.info("\nResidual Statistics:")
        logger.info(f"Mean: {resid_stats['mean']:.4f}")
        logger.info(f"Std: {resid_stats['std']:.4f}")
        logger.info(f"Min: {resid_stats['min']:.4f}")
        logger.info(f"Max: {resid_stats['max']:.4f}")
    
    if 'shapiro_wilk' in diagnostics and diagnostics['shapiro_wilk'] is not None:
        logger.info(f"\nShapiro-Wilk test for normality:")
        if 'statistic' in diagnostics['shapiro_wilk'] and diagnostics['shapiro_wilk']['statistic'] is not None:
            logger.info(f"Statistic: {diagnostics['shapiro_wilk']['statistic']:.4f}")
        if 'pvalue' in diagnostics['shapiro_wilk'] and diagnostics['shapiro_wilk']['pvalue'] is not None:
            logger.info(f"p-value: {diagnostics['shapiro_wilk']['pvalue']:.4f}")
            
    if 'ljung_box' in diagnostics and diagnostics['ljung_box'] is not None:
        logger.info("\nLjung-Box test for autocorrelation:")
        if all(key in diagnostics['ljung_box'] for key in ['lags', 'statistic', 'pvalue']):
            for i, lag in enumerate(diagnostics['ljung_box']['lags']):
                stat = diagnostics['ljung_box']['statistic'][i]
                pval = diagnostics['ljung_box']['pvalue'][i]
                if stat is not None and pval is not None:
                    logger.info(f"Lag {lag}:")
                    logger.info(f"Statistic: {stat:.4f}")
                    logger.info(f"p-value: {pval:.4f}")
                    
    if 'aic' in diagnostics and diagnostics['aic'] is not None:
        logger.info(f"\nInformation Criteria:")
        logger.info(f"AIC: {diagnostics['aic']:.4f}")
        if 'bic' in diagnostics and diagnostics['bic'] is not None:
            logger.info(f"BIC: {diagnostics['bic']:.4f}")

class FixedFeatureSARIMAX(EnhancedSARIMAX):
    """Modified SARIMAX model that uses fixed features without secondary filtering."""
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        freq: str = 'ME',
        fixed_features: List[str] = None
    ) -> None:
        """Fit the SARIMAX model with fixed features and orders.
        
        Args:
            X: Feature matrix
            y: Target variable
            freq: Time series frequency
            fixed_features: List of features to use (bypasses feature selection)
        """
        # Ensure datetime index with proper frequency
        if not isinstance(X.index, pd.DatetimeIndex):
            X.index = pd.to_datetime(X.index)
        if not isinstance(y.index, pd.DatetimeIndex):
            y.index = pd.to_datetime(y.index)
            
        # Reindex with proper frequency
        date_range = pd.date_range(start=X.index.min(), end=X.index.max(), freq=freq)
        X = X.reindex(date_range)
        y = y.reindex(date_range)
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        
        # Use fixed features if provided instead of feature selection
        if fixed_features:
            self.selected_features = fixed_features
            logger.info(f"Using fixed feature set ({len(fixed_features)} features):")
            for feature in fixed_features:
                logger.info(f"- {feature}")
        else:
            # Fall back to feature selection if no fixed features provided
            self.selected_features = self.select_significant_features(X_processed, y_processed)
            
        X_filtered = X_processed[self.selected_features] if self.selected_features else None
        
        # Use fixed model orders instead of grid search
        self.best_order = (1, 1, 1)  # p, d, q
        self.best_seasonal_order = (1, 0, 1, 12)  # P, D, Q, m
        
        logger.info(f"Using fixed model orders:")
        logger.info(f"ARIMA({self.best_order[0]},{self.best_order[1]},{self.best_order[2]})")
        logger.info(f"Seasonal({self.best_seasonal_order[0]},{self.best_seasonal_order[1]},{self.best_seasonal_order[2]})_{self.best_seasonal_order[3]}")
        
        # Fit final model with fixed orders
        self.model = SARIMAX(
            y_processed,
            exog=X_filtered,
            order=self.best_order,
            seasonal_order=self.best_seasonal_order,
            freq=freq,
            enforce_stationarity=False,  # Allow non-stationary
            enforce_invertibility=False  # Allow non-invertible
        )
        
        # Fit with improved optimization settings
        self.result = self.model.fit(
            disp=False,
            maxiter=1000,
            method='lbfgs',
            optim_score='harvey',
            cov_type='robust'
        )
        
        # Log model summary
        logger.info("SARIMAX model summary:")
        logger.info(f"AIC: {self.result.aic:.4f}")
        logger.info(f"BIC: {self.result.bic:.4f}")

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
        
        # Define the 5 low-VIF features to use
        fixed_features = [
            'Henry_Hub_Spot',
            'Newcastle_FOB_6000_NAR_roll_std_3',
            'Newcastle_FOB_6000_NAR_mom_change',
            'Newcastle_FOB_6000_NAR_mom_change_3m_avg',
            'Baltic_Dry_Index_scaled'
        ]
        
        logger.info(f"\nUsing fixed set of 5 low-VIF features:")
        for feature in fixed_features:
            logger.info(f"- {feature}")
            
        # Initialize model with fixed parameters
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = FixedFeatureSARIMAX(
            config,
            seasonal=True,
            m=12
        )
        logger.info("\nFitting SARIMAX(1,1,1)(1,0,1,12) model with 5 fixed low-VIF features...")
        
        # Fit model with fixed features
        model.fit(X_train, y_train, freq='ME', fixed_features=fixed_features)
        
        # Get diagnostics
        diagnostics = model.get_diagnostics()
        log_diagnostic_results(diagnostics)
        
        # Save model summary
        os.makedirs(os.path.join(PROJECT_ROOT, 'reports'), exist_ok=True)
        save_model_summary(
            model,
            os.path.join(PROJECT_ROOT, 'reports/sarimax_fixed_features_summary.txt')
        )
        
        # Plot diagnostics
        plot_model_diagnostics(
            residuals=diagnostics['residuals'],
            config=config
        )
        
        # Generate and evaluate predictions using fixed features
        train_pred, _ = model.predict(X_train[fixed_features])
        test_pred, _ = model.predict(X_test[fixed_features])
        
        # Ensure predictions are aligned with actual values
        if train_pred is not None and test_pred is not None:
            # Ensure index alignment
            train_pred.index = X_train.index
            test_pred.index = X_test.index
            
            # Log prediction shapes for debugging
            logger.info("\nPrediction shapes:")
            logger.info(f"Train predictions: {train_pred.shape}")
            logger.info(f"Test predictions: {test_pred.shape}")
            logger.info(f"Train actual: {y_train.shape}")
            logger.info(f"Test actual: {y_test.shape}")
            
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
                    
            # Save metrics to CSV
            metrics_df = pd.DataFrame({
                'Metric': ['R2', 'RMSE', 'MAE', 'MAPE'],
                'Train': [
                    metrics['train'].get('r2', np.nan),
                    metrics['train'].get('rmse', np.nan),
                    metrics['train'].get('mae', np.nan),
                    metrics['train'].get('mape', np.nan)
                ],
                'Test': [
                    metrics['test'].get('r2', np.nan),
                    metrics['test'].get('rmse', np.nan),
                    metrics['test'].get('mae', np.nan),
                    metrics['test'].get('mape', np.nan)
                ]
            })
            metrics_df.to_csv(os.path.join(PROJECT_ROOT, 'reports/sarimax_fixed_features_metrics.csv'), index=False)
            logger.info(f"\nMetrics saved to reports/sarimax_fixed_features_metrics.csv")
            
        else:
            logger.error("Failed to generate predictions. Check model fitting and prediction steps.")
            
    except Exception as e:
        logger.error(f"Error in SARIMAX evaluation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()