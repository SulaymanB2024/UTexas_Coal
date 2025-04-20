import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import logging
import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.utils.logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all forecasting models."""
    
    def __init__(self):
        """Initialize the base model."""
        pass
        
    def fit(self, X, y):
        """Fit the model to training data.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            
        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement fit()")
        
    def predict(self, X):
        """Generate predictions.
        
        Args:
            X (pd.DataFrame): Feature matrix for prediction
            
        Returns:
            tuple: (predictions, prediction_std)
        """
        raise NotImplementedError("Subclasses must implement predict()")
        
    def prepare_data(self, data, target_transform=None):
        """Prepare data for modeling by splitting features and target.
        
        Args:
            data (pd.DataFrame): Input dataframe
            target_transform (str): Optional transformation to apply to target
                                 (e.g., '_log' for log-transformed target)
        
        Returns:
            tuple: X (features) and y (target) DataFrames
        """
        if target_transform:
            target_col = f"{self.target_name}{target_transform}"
        else:
            target_col = self.target_name
            
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not found in data")
            
        y = data[target_col]
        
        # Remove target and its transformations from features
        X = data.drop([col for col in data.columns 
                      if col.startswith(self.target_name)], axis=1)
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        return X, y
        
    def create_cv_folds(self, X):
        """Create time series cross-validation folds.
        
        Args:
            X (pd.DataFrame): Feature DataFrame
            
        Returns:
            TimeSeriesSplit: Cross-validation splitter
        """
        return TimeSeriesSplit(
            n_splits=self.config['modeling']['cv_folds'],
            test_size=self.config['modeling']['cv_rolling_test_size']
        )
        
    def evaluate(self, y_true, y_pred, prefix=''):
        """Calculate and log model performance metrics.
        
        Args:
            y_true (array-like): True target values
            y_pred (array-like): Predicted target values
            prefix (str): Optional prefix for metric names (e.g., 'train_' or 'test_')
            
        Returns:
            dict: Dictionary of metrics
        """
        # Convert to numpy arrays to avoid timestamp comparison issues
        y_true_values = pd.Series(y_true).values
        y_pred_values = pd.Series(y_pred).values
        
        metrics = {
            f'{prefix}rmse': np.sqrt(mean_squared_error(y_true_values, y_pred_values)),
            f'{prefix}mae': mean_absolute_error(y_true_values, y_pred_values),
            f'{prefix}r2': r2_score(y_true_values, y_pred_values)
        }
        
        # Calculate MAPE only if no zero values in y_true
        if not np.any(y_true_values == 0):
            metrics[f'{prefix}mape'] = np.mean(np.abs((y_true_values - y_pred_values) / y_true_values)) * 100
        else:
            metrics[f'{prefix}mape'] = np.nan
            logger.warning("MAPE could not be calculated due to zero values in actual data")
        
        # Calculate directional accuracy using numpy arrays
        y_true_diff = np.diff(y_true_values)
        y_pred_diff = np.diff(y_pred_values)
        direction_correct = np.sum(np.sign(y_true_diff) == np.sign(y_pred_diff))
        metrics[f'{prefix}dir_acc'] = direction_correct / len(y_true_diff) * 100
        
        # Log metrics
        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")
            
        return metrics
        
    def save_predictions(self, y_true, y_pred, filepath):
        """Save predictions to CSV for analysis.
        
        Args:
            y_true (pd.Series): True target values
            y_pred (pd.Series): Predicted values
            filepath (str): Path to save predictions
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create predictions DataFrame
            predictions = pd.DataFrame({
                'y_true': y_true,
                'y_pred': y_pred
            })
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            predictions.to_csv(filepath)
            logger.info(f"Successfully saved predictions to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            return False
            
    def get_forecast_dates(self, last_date, periods):
        """Generate future dates for forecasting.
        
        Args:
            last_date (pd.Timestamp): Last date in the training data
            periods (int): Number of periods to forecast
            
        Returns:
            pd.DatetimeIndex: Index of forecast dates
        """
        # Assuming monthly frequency
        return pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='M'
        )
        
    def _validate_data(self, X, y):
        """Validate input data before modeling.
        
        Args:
            X (pd.DataFrame): Feature DataFrame
            y (pd.Series): Target series
            
        Returns:
            bool: True if validation passes
        """
        if X is None or y is None:
            logger.error("Input data is None")
            return False
            
        if len(X) != len(y):
            logger.error(f"Feature and target lengths don't match: {len(X)} vs {len(y)}")
            return False
            
        if X.isnull().any().any():
            logger.error("Features contain missing values")
            return False
            
        if y.isnull().any():
            logger.error("Target contains missing values")
            return False
            
        return True