import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank, VECM
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
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
            
    def prepare_features(self, X):
        """Prepare features for ARIMAX model."""
        try:
            # Add seasonality features
            X = self._add_seasonality(X)
            
            # Select important predictors
            essential_features = []
            
            # Add log-transformed features
            log_features = [col for col in X.columns if '_log' in col]
            essential_features.extend(log_features)
            
            # Add lagged features (exclude target lags beyond order 3)
            lag_features = [col for col in X.columns if '_lag_' in col 
                          and (not col.startswith('Newcastle_FOB_6000_NAR') 
                               or any(f'_lag_{i}' in col for i in [1,2,3]))]
            essential_features.extend(lag_features)
            
            # Add scaled features
            scaled_features = [col for col in X.columns if '_scaled' in col]
            essential_features.extend(scaled_features)
            
            # Add seasonality
            essential_features.extend(['month', 'quarter'])
            
            # Remove duplicates while preserving order
            essential_features = list(dict.fromkeys(essential_features))
            
            # Store feature columns for prediction
            self.feature_columns = essential_features
            
            # Return cleaned features
            X_clean = X[essential_features].copy()
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
            X_clean = X_clean.ffill().bfill()
            
            return X_clean
            
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