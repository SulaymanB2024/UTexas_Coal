"""Evaluate enhanced SARIMAX model and compare with previous ARIMAX results."""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')

from src.utils.logging_config import setup_logging, load_config
from src.modeling.sarimax_model import EnhancedSARIMAX
from src.modeling.ts_models import TSModels
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

def analyze_multicollinearity(X: pd.DataFrame, config: Dict) -> List[str]:
    """Analyze multicollinearity in features using VIF and return selected features.
    
    Args:
        X: Feature matrix
        config: Configuration dictionary
        
    Returns:
        List of selected feature names with VIF below threshold
    """
    logger.info("\nMulticollinearity Analysis:")
    
    # Handle missing and infinite values for VIF calculation
    X_clean = X.copy()
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    
    # Log NaN handling
    nan_counts = X_clean.isna().sum()
    if nan_counts.any():
        logger.info("\nHandling missing values for VIF calculation:")
        for col in X_clean.columns:
            count = nan_counts[col]
            if count > 0:
                logger.info(f"Imputing {count} missing values in {col}")
    
    # Use median imputation for VIF calculation
    X_clean = X_clean.fillna(X_clean.median())
    
    # Initialize model with default VIF threshold
    model = TSModels(config)
    
    # Calculate initial VIF scores
    initial_vif = model.calculate_vif(X_clean)
    if initial_vif is not None:
        logger.info("\nInitial VIF Scores:")
        logger.info("\n" + str(initial_vif))
        
        # Identify features with high VIF
        high_vif = initial_vif[initial_vif['VIF'] > model.vif_threshold]
        if not high_vif.empty:
            logger.info(f"\nFeatures with VIF > {model.vif_threshold}:")
            logger.info("\n" + str(high_vif))
        
        # Perform iterative VIF selection
        X_selected, removed_features = model.iterative_vif_selection(X_clean)
        
        if removed_features:
            logger.info("\nFeatures removed due to high VIF:")
            for feature, vif in removed_features:
                logger.info(f"{feature}: {vif:.2f}")
        
        # Calculate final VIF scores
        final_vif = model.calculate_vif(X_selected)
        if final_vif is not None:
            logger.info("\nFinal VIF Scores after feature selection:")
            logger.info("\n" + str(final_vif))
            
            # Save VIF analysis results
            os.makedirs(os.path.join(PROJECT_ROOT, 'reports'), exist_ok=True)
            results_path = os.path.join(PROJECT_ROOT, 'reports/vif_analysis_results.csv')
            initial_vif.to_csv(results_path)
            logger.info(f"\nVIF analysis results saved to: {results_path}")
            
            return X_selected.columns.tolist()
            
    return []

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

def main():
    """Main execution function."""
    # Load config and setup logging
    config = load_config(CONFIG_PATH)
    setup_logging(config)
    
    try:
        # Load data using existing function
        train_data, test_data = load_data()
        
        target_var = config['variables']['target_variable']
        
        # Separate features and target
        X_train = train_data.drop(columns=[target_var])
        y_train = train_data[target_var]
        X_test = test_data.drop(columns=[target_var])
        y_test = test_data[target_var]
        
        # Analyze multicollinearity and get selected features
        logger.info("Analyzing feature multicollinearity...")
        selected_features = analyze_multicollinearity(X_train, config)
        
        if not selected_features:
            logger.error("No features selected after VIF analysis. Check VIF thresholds and feature correlations.")
            return
            
        logger.info(f"\nSelected {len(selected_features)} features for modeling:")
        for feature in selected_features:
            logger.info(f"- {feature}")
            
        # Use only selected features for modeling
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Initialize and fit model with selected features
        model = EnhancedSARIMAX(config)
        logger.info("\nFitting Enhanced SARIMAX model with selected features...")
        
        # Fit model with selected features
        model.fit(X_train_selected, y_train, freq='ME')
        
        # Get diagnostics
        diagnostics = model.get_diagnostics()
        log_diagnostic_results(diagnostics)
        
        # Plot diagnostics
        plot_model_diagnostics(
            residuals=diagnostics['residuals'],
            config=config
        )
        
        # Generate and evaluate predictions using selected features
        train_pred, _ = model.predict(X_train_selected)
        test_pred, _ = model.predict(X_test_selected)
        
        # Ensure predictions are aligned with actual values
        if train_pred is not None and test_pred is not None:
            # Ensure index alignment
            train_pred.index = X_train_selected.index
            test_pred.index = X_test_selected.index
            
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
        else:
            logger.error("Failed to generate predictions. Check model fitting and prediction steps.")
            
    except Exception as e:
        logger.error(f"Error in SARIMAX evaluation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()