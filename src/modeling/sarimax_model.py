"""Enhanced SARIMAX model implementation for coal price forecasting."""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from itertools import product
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

logger = logging.getLogger(__name__)

class EnhancedSARIMAX:
    """Implements an enhanced SARIMAX model with information criteria-based order selection."""
    
    def __init__(
        self,
        config: Dict,
        seasonal: bool = True,
        m: int = 12,
        max_p: int = 2,
        max_q: int = 2,
        max_P: int = 1,
        max_Q: int = 1,
        max_d: int = 1,
        max_D: int = 1
    ):
        """Initialize the model.
        
        Args:
            config: Configuration dictionary
            seasonal: Whether to include seasonal components
            m: Seasonal period
            max_p, max_q: Maximum AR and MA orders
            max_P, max_Q: Maximum seasonal AR and MA orders
            max_d, max_D: Maximum differencing orders
        """
        self.config = config
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_d = max_d
        self.max_D = max_D
        self.model = None
        self.best_order = None
        self.best_seasonal_order = None
        self.result = None
        
    def preprocess_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Preprocess data by handling missing values and outliers.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            fit: Whether this is a fit operation (True) or transform operation (False)
            
        Returns:
            Tuple of (processed X, processed y if y was provided)
        """
        if fit:
            self.feature_medians = X.median()
            if y is not None:
                self.target_median = y.median()
        
        # Handle missing and infinite values in features
        X_processed = X.copy()
        for col in X_processed.columns:
            mask = ~np.isfinite(X_processed[col])
            if mask.any():
                logger.info(f"Replacing {mask.sum()} invalid values in {col}")
                X_processed.loc[mask, col] = self.feature_medians[col]
        
        # Handle missing and infinite values in target
        y_processed = None
        if y is not None:
            y_processed = y.copy()
            mask = ~np.isfinite(y_processed)
            if mask.any():
                logger.info(f"Replacing {mask.sum()} invalid values in target")
                y_processed[mask] = self.target_median
        
        return X_processed, y_processed
        
    def check_stationarity(self, data: np.ndarray) -> Tuple[bool, int]:
        """Determine appropriate differencing order using ADF and KPSS tests."""
        # ADF Test
        adf_result = adfuller(data)
        adf_pvalue = adf_result[1]
        
        # KPSS Test
        kpss_result = kpss(data)
        kpss_pvalue = kpss_result[1]
        
        logger.info("Stationarity tests:")
        logger.info(f"ADF p-value: {adf_pvalue:.4f}")
        logger.info(f"KPSS p-value: {kpss_pvalue:.4f}")
        
        # Decision logic
        if adf_pvalue > 0.05 and kpss_pvalue <= 0.05:
            return False, 1
        elif adf_pvalue <= 0.05 and kpss_pvalue > 0.05:
            return True, 0
        else:
            return False, 1
            
    def select_significant_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        significance_level: float = 0.10
    ) -> List[str]:
        """Select statistically significant features using OLS regression."""
        import statsmodels.api as sm
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        
        # Fit OLS model
        X_with_const = sm.add_constant(X_processed)
        ols_model = sm.OLS(y_processed, X_with_const).fit()
        
        # Get significant features
        significant_features = []
        for i, pvalue in enumerate(ols_model.pvalues[1:]):
            if pvalue < significance_level:
                significant_features.append(X.columns[i])
                
        logger.info(f"Selected {len(significant_features)} significant features:")
        for feat in significant_features:
            logger.info(f"- {feat}")
            
        return significant_features
        
    def _evaluate_model(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
        freq: str
    ) -> float:
        """Evaluate a SARIMAX model with given orders using AIC."""
        try:
            # Preprocess data
            X_processed, y_processed = self.preprocess_data(X, y, fit=False)
            
            model = SARIMAX(
                y_processed,
                exog=X_processed,
                order=order,
                seasonal_order=seasonal_order,
                freq=freq
            )
            result = model.fit(disp=False)
            return result.aic
        except:
            return np.inf
            
    def _grid_search_orders(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        d: int,
        freq: str
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """Perform grid search for optimal SARIMAX orders."""
        best_aic = np.inf
        best_order = None
        best_seasonal_order = None
        
        # Generate candidate orders
        p = range(self.max_p + 1)
        q = range(self.max_q + 1)
        P = range(self.max_P + 1) if self.seasonal else [0]
        Q = range(self.max_Q + 1) if self.seasonal else [0]
        D = range(self.max_D + 1) if self.seasonal else [0]
        
        combinations = list(product(p, q, P, Q, D))
        total = len(combinations)
        
        logger.info(f"Grid searching over {total} model combinations...")
        
        for i, (p, q, P, Q, D) in enumerate(combinations):
            if i % 10 == 0:
                logger.info(f"Evaluated {i}/{total} combinations...")
                
            order = (p, d, q)
            seasonal_order = (P, D, Q, self.m) if self.seasonal else (0, 0, 0, 0)
            
            aic = self._evaluate_model(y, X, order, seasonal_order, freq)
            
            if aic < best_aic:
                best_aic = aic
                best_order = order
                best_seasonal_order = seasonal_order
                
        logger.info(f"Best model - AIC: {best_aic:.4f}")
        logger.info(f"Non-seasonal order (p,d,q): {best_order}")
        logger.info(f"Seasonal order (P,D,Q,m): {best_seasonal_order}")
        
        return best_order, best_seasonal_order
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        freq: str = 'ME'
    ) -> None:
        """Fit the SARIMAX model with optimal order selection."""
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
        
        # Select significant features
        self.selected_features = self.select_significant_features(X_processed, y_processed)
        X_filtered = X_processed[self.selected_features] if self.selected_features else None
        
        # Check stationarity and get differencing order
        _, d = self.check_stationarity(y_processed.values)
        
        # Grid search for optimal orders
        self.best_order, self.best_seasonal_order = self._grid_search_orders(
            y_processed,
            X_filtered,
            d,
            freq
        )
        
        # Fit final model with optimal orders
        self.model = SARIMAX(
            y_processed,
            exog=X_filtered,
            order=self.best_order,
            seasonal_order=self.best_seasonal_order,
            freq=freq,
            enforce_stationarity=True,  # Changed to True to ensure stationarity
            enforce_invertibility=True   # Changed to True to ensure invertibility
        )
        
        # Fit with improved optimization settings
        self.result = self.model.fit(
            disp=False,
            maxiter=1000,  # Increased maximum iterations
            method='lbfgs',  # Changed to L-BFGS-B optimizer
            optim_score='harvey',  # Use Harvey's score to improve numerical stability
            cov_type='robust'  # Use robust covariance estimation
        )
        
        # Log model summary
        logger.info("SARIMAX model summary:")
        logger.info(f"AIC: {self.result.aic:.4f}")
        logger.info(f"BIC: {self.result.bic:.4f}")
        
    def predict(self, X_train=None, X_test=None):
        """
        Generate predictions for both training and test sets.
        
        Args:
            X_train (pd.DataFrame): Training features for in-sample predictions
            X_test (pd.DataFrame): Test features for out-of-sample predictions
        
        Returns:
            tuple: (train_predictions, test_predictions)
        """
        if not hasattr(self, 'result'):
            raise ValueError("Model must be fitted before making predictions")
        
        train_pred = None
        test_pred = None
        
        # Generate in-sample predictions for training data
        if X_train is not None:
            X_train_filtered = X_train[self.selected_features] if self.selected_features else None
            train_pred = self.result.get_prediction(
                start=0,
                end=len(X_train) - 1,
                exog=X_train_filtered
            ).predicted_mean
        
        # Generate out-of-sample predictions for test data
        if X_test is not None:
            X_test_filtered = X_test[self.selected_features] if self.selected_features else None
            
            # Calculate the correct start and end points for test predictions
            start = len(X_train) if X_train is not None else 0
            end = start + len(X_test) - 1
            
            # Ensure exogenous variables match the forecast period
            if X_test_filtered is not None and len(X_test_filtered) != (end - start + 1):
                raise ValueError(f"Length of test exogenous variables ({len(X_test_filtered)}) must match forecast period ({end - start + 1})")
            
            test_pred = self.result.forecast(
                steps=len(X_test),
                exog=X_test_filtered
            )
        
        return train_pred, test_pred
    
    def forecast(self, steps, exog=None):
        """
        Generate forecasts for future periods.
        
        Args:
            steps (int): Number of steps to forecast
            exog (pd.DataFrame): Exogenous variables for the forecast period
        
        Returns:
            pd.Series: Forecast values
        """
        if not hasattr(self, 'model_fit'):
            raise ValueError("Model must be fitted before forecasting")
            
        if self.exog_columns and exog is None:
            raise ValueError("Exogenous variables required for forecasting")
            
        if self.exog_columns and len(exog) != steps:
            raise ValueError(f"Length of exogenous variables ({len(exog)}) must match forecast steps ({steps})")
        
        forecast = self.model_fit.get_forecast(
            steps=steps,
            exog=exog if self.exog_columns else None
        )
        
        return forecast.predicted_mean
            
    def get_diagnostics(self):
        """Get model diagnostic information.
        
        Returns:
            Dictionary containing diagnostic test results
        """
        residuals = self.result.resid
        
        # Shapiro-Wilk test for normality
        _, sw_pvalue = stats.shapiro(residuals)
        
        # Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
        
        return {
            'shapiro_wilk': {
                'statistic': None,  # Not storing test statistic for now
                'pvalue': sw_pvalue
            },
            'ljung_box': {
                'statistic': lb_test['lb_stat'].values.tolist(),
                'pvalue': lb_test['lb_pvalue'].values.tolist(),
                'lags': [10, 20, 30]
            },
            'residuals': residuals,
            'aic': self.result.aic,
            'bic': self.result.bic
        }