import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank, VECM
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.utils.logging_config import setup_logging
from src.modeling.base_model import BaseModel

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

class TimeSeriesModel(BaseModel):
    """Implements time series econometric models (VECM/GARCH) for coal price forecasting."""
    
    def __init__(self, config):
        """Initialize the time series model.
        
        Args:
            config (dict): Configuration dictionary containing model parameters
        """
        super().__init__(config)
        self.vecm_model = None
        self.garch_model = None
        self.selected_lags = None
        self.coint_rank = None
        
    def select_vecm_order(self, data):
        """Select optimal VECM lag order using information criteria.
        
        Args:
            data (pd.DataFrame): Input data for VECM
            
        Returns:
            int: Selected number of lags
        """
        try:
            # Use VAR lag order selection for VECM
            lag_order = select_order(
                data,
                maxlags=12,  # Maximum 12 months of lags
                deterministic="ci"  # Constant and intercept
            )
            
            # Get selected lags from AIC
            selected_lags = lag_order.aic
            logger.info(f"Selected VECM lag order: {selected_lags}")
            return selected_lags
            
        except Exception as e:
            logger.error(f"Error in VECM lag selection: {e}")
            return None
            
    def determine_cointegration_rank(self, data):
        """Determine cointegration rank using Johansen procedure.
        
        Args:
            data (pd.DataFrame): Input data for cointegration test
            
        Returns:
            int: Cointegration rank
        """
        try:
            # Perform Johansen cointegration test
            coint_test = select_coint_rank(
                data,
                det_order=1,  # Linear trend
                k_ar_diff=self.selected_lags,
                method='trace'
            )
            
            rank = coint_test.rank
            logger.info(f"Selected cointegration rank: {rank}")
            return rank
            
        except Exception as e:
            logger.error(f"Error in cointegration rank selection: {e}")
            return None
            
    def fit_vecm(self, data):
        """Fit VECM model to the data.
        
        Args:
            data (pd.DataFrame): Input data for VECM fitting
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Select optimal lag order if not already done
            if self.selected_lags is None:
                self.selected_lags = self.select_vecm_order(data)
                
            if self.selected_lags is None:
                return False
                
            # Determine cointegration rank if not already done
            if self.coint_rank is None:
                self.coint_rank = self.determine_cointegration_rank(data)
                
            if self.coint_rank is None:
                return False
                
            # Fit VECM model
            self.vecm_model = VECM(
                data,
                k_ar_diff=self.selected_lags,
                deterministic="ci",
                coint_rank=self.coint_rank
            ).fit()
            
            logger.info("Successfully fitted VECM model")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting VECM model: {e}")
            return False
            
    def fit_garch(self, residuals):
        """Fit GARCH model to VECM residuals.
        
        Args:
            residuals (pd.Series): Residuals from VECM model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Fit GARCH(1,1) model
            self.garch_model = arch_model(
                residuals,
                vol='Garch',
                p=1,
                q=1,
                mean='Zero',
                dist='normal'
            ).fit(disp='off')
            
            logger.info("Successfully fitted GARCH model")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            return False
            
    def forecast_vecm(self, steps=3):
        """Generate VECM forecasts.
        
        Args:
            steps (int): Number of steps ahead to forecast
            
        Returns:
            pd.DataFrame: Forecasted values
        """
        if self.vecm_model is None:
            logger.error("VECM model not fitted. Call fit_vecm() first.")
            return None
            
        try:
            # Generate forecasts
            forecasts = self.vecm_model.forecast(
                steps=steps,
                alpha=0.05  # 95% confidence intervals
            )
            
            return pd.DataFrame(
                forecasts[0],  # Point forecasts
                columns=self.vecm_model.names,
                index=self.get_forecast_dates(
                    self.vecm_model.data.dates[-1],
                    steps
                )
            )
            
        except Exception as e:
            logger.error(f"Error generating VECM forecasts: {e}")
            return None
            
    def forecast_volatility(self, steps=3):
        """Generate volatility forecasts using GARCH model.
        
        Args:
            steps (int): Number of steps ahead to forecast
            
        Returns:
            pd.Series: Forecasted volatilities
        """
        if self.garch_model is None:
            logger.error("GARCH model not fitted. Call fit_garch() first.")
            return None
            
        try:
            # Generate volatility forecasts
            forecasts = self.garch_model.forecast(horizon=steps)
            
            return pd.Series(
                forecasts.variance.values[-1],  # Last row contains the forecasts
                index=self.get_forecast_dates(
                    self.garch_model.last_date,
                    steps
                )
            )
            
        except Exception as e:
            logger.error(f"Error generating volatility forecasts: {e}")
            return None
            
    def fit(self, X, y):
        """Fit both VECM and GARCH models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_data(X, y):
            return False
            
        try:
            # Combine features and target for VECM
            data = pd.concat([y, X], axis=1)
            
            # Fit VECM
            if not self.fit_vecm(data):
                return False
                
            # Get residuals from target equation and fit GARCH
            target_residuals = self.vecm_model.resid[self.target_name]
            if not self.fit_garch(target_residuals):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in model fitting: {e}")
            return False
            
    def predict(self, X=None, steps=3):
        """Generate price and volatility forecasts.
        
        Args:
            X (pd.DataFrame): Optional features for the forecast period
            steps (int): Number of steps to forecast
            
        Returns:
            tuple: (price_forecasts, volatility_forecasts) DataFrames
        """
        try:
            price_forecasts = self.forecast_vecm(steps=steps)
            vol_forecasts = self.forecast_volatility(steps=steps)
            
            if price_forecasts is not None:
                # Extract only the target variable forecasts
                price_forecasts = price_forecasts[self.target_name]
                
            return price_forecasts, vol_forecasts
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {e}")
            return None, None

class TSModels:
    """Time Series Models for coal price forecasting."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.d = None
        self.feature_columns = None  # Store feature columns for prediction
        self.vif_threshold = 10.0  # Default VIF threshold
        
    def _add_seasonality(self, X):
        """Add month and quarter features if not present."""
        X = X.copy()
        if isinstance(X.index, pd.DatetimeIndex):
            if 'month' not in X.columns:
                X['month'] = X.index.month
            if 'quarter' not in X.columns:
                X['quarter'] = X.index.quarter
        return X
        
    def test_stationarity(self, series):
        """Test time series for stationarity using ADF and KPSS tests."""
        try:
            results = {}
            
            # ADF Test
            adf_result = adfuller(series, regression='ct')
            results['adf'] = {
                'pvalue': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            }
            
            # KPSS Test
            kpss_result = kpss(series, regression='ct', nlags='auto')
            results['kpss'] = {
                'pvalue': kpss_result[1],
                'is_stationary': kpss_result[1] >= 0.05
            }
            
            # Determine order of integration (d)
            if results['adf']['is_stationary'] and results['kpss']['is_stationary']:
                self.d = 0
            else:
                self.d = 1  # Use first difference based on our analysis
                
            logger.info(f"Stationarity tests:")
            logger.info(f"ADF p-value: {results['adf']['pvalue']:.4f}")
            logger.info(f"KPSS p-value: {results['kpss']['pvalue']:.4f}")
            logger.info(f"Using order of integration (d): {self.d}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in stationarity testing: {e}")
            return None
            
    def calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate VIF for each feature in the dataset.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            pd.DataFrame: DataFrame with VIF scores for each feature
        """
        try:
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                              for i in range(X.shape[1])]
            return vif_data.sort_values('VIF', ascending=False)
            
        except Exception as e:
            logger.error(f"Error calculating VIF scores: {e}")
            return None
            
    def iterative_vif_selection(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Iteratively remove features with highest VIF until all are below threshold.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: (Selected features DataFrame, List of removed features)
        """
        try:
            features = X.columns.tolist()
            removed_features = []
            
            while True:
                if len(features) < 2:
                    break
                    
                vif = self.calculate_vif(X[features])
                if vif is None:
                    break
                    
                max_vif = vif['VIF'].max()
                if max_vif < self.vif_threshold:
                    break
                    
                feature_to_remove = vif.loc[vif['VIF'].idxmax(), 'Feature']
                features.remove(feature_to_remove)
                removed_features.append((feature_to_remove, max_vif))
                
                logger.info(f"Removed feature {feature_to_remove} with VIF: {max_vif:.2f}")
                
            return X[features], removed_features
            
        except Exception as e:
            logger.error(f"Error in VIF-based feature selection: {e}")
            return X, []
            
    def prepare_features(self, X):
        """Prepare features for ARIMAX model."""
        try:
            # Add seasonality features
            X = self._add_seasonality(X)
            
            # Select features and handle VIF
            X_prepared = X.copy()
            X_prepared = X_prepared.replace([np.inf, -np.inf], np.nan)
            X_prepared = X_prepared.ffill().bfill()
            
            # Calculate initial VIF scores
            initial_vif = self.calculate_vif(X_prepared)
            if initial_vif is not None:
                logger.info("Initial VIF scores:")
                logger.info("\n" + str(initial_vif))
                
                # Perform VIF-based feature selection
                X_selected, removed_features = self.iterative_vif_selection(X_prepared)
                if removed_features:
                    logger.info("\nRemoved features due to high VIF:")
                    for feature, vif in removed_features:
                        logger.info(f"{feature}: {vif:.2f}")
                
                # Store selected feature columns
                self.feature_columns = X_selected.columns.tolist()
                return X_selected
                
            return X_prepared
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
            
    def fit(self, X, y):
        """Fit ARIMAX model."""
        try:
            # Test stationarity and determine d
            self.test_stationarity(y)
            
            # Prepare features
            X_prepared = self.prepare_features(X)
            if X_prepared is None:
                return False
                
            # Use ARIMAX with order based on stationarity testing
            self.model = ARIMA(
                y,
                order=(2, self.d, 1),  # p=2, d from testing, q=1
                exog=X_prepared,
                freq='ME'  # Using 'ME' for month end frequency
            ).fit(
                method='statespace',  # Use state space method for better convergence
                cov_type='robust'     # Robust covariance estimation
            )
            
            logger.info(f"ARIMAX model summary:")
            logger.info(f"AIC: {self.model.aic:.4f}")
            logger.info(f"BIC: {self.model.bic:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fitting ARIMAX model: {e}")
            return False
            
    def predict(self, X):
        """Generate predictions with uncertainty estimates.
        
        Args:
            X (pd.DataFrame): Input features for prediction
            
        Returns:
            tuple: (predictions, prediction_std) containing point predictions 
                  and their standard deviations
        """
        try:
            if self.model is None:
                logger.error("Model not fitted. Call fit() first.")
                return None, None
                
            if self.feature_columns is None:
                logger.error("Feature columns not set. Call fit() first.")
                return None, None
            
            # Prepare features for prediction
            X_pred = self.prepare_features(X)
            if X_pred is None:
                return None, None
            
            # Generate forecasts with prediction intervals
            forecast_result = self.model.get_forecast(steps=len(X), exog=X_pred)
            predictions = forecast_result.predicted_mean
            
            # Get prediction intervals
            conf_int = forecast_result.conf_int(alpha=0.32)  # ~1 std dev
            prediction_std = (conf_int.iloc[:, 1] - conf_int.iloc[:, 0]) / 2
            
            return predictions, prediction_std
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return None, None

if __name__ == '__main__':
    from src.utils.logging_config import load_config, CONFIG_PATH
    
    # Load configuration
    config = load_config(CONFIG_PATH)
    if config is None:
        logger.critical("Failed to load configuration. Exiting.")
        sys.exit(1)
        
    # Example usage (assuming data is available)
    model = TimeSeriesModel(config)