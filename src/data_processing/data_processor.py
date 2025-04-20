import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os
import sys

# Add root directory to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from src.utils.logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data cleaning, transformation, and feature engineering for the coal price forecasting model."""
    
    def __init__(self, config):
        """Initialize the DataProcessor with configuration settings.
        
        Args:
            config (dict): Configuration dictionary containing data paths and parameters
        """
        self.config = config
        self.data = None
        
    def load_data(self, filepath):
        """Load data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.data = pd.read_csv(filepath)
            self.data[self.config['variables']['date_column']] = pd.to_datetime(
                self.data[self.config['variables']['date_column']]
            )
            self.data.set_index(self.config['variables']['date_column'], inplace=True)
            logger.info(f"Successfully loaded data from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return False
            
    def clean_data(self):
        """Clean the loaded data by handling missing values and outliers."""
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return False
            
        try:
            # Handle missing values
            for col in self.data.columns:
                missing_count = self.data[col].isnull().sum()
                if missing_count > 0:
                    logger.warning(f"Found {missing_count} missing values in {col}")
                    
                    # Use linear interpolation for price and numeric data
                    if col == self.config['variables']['target_variable'] or \
                       col in self.config['variables']['predictors']['core']:
                        self.data[col] = self.data[col].interpolate(method='linear')
                    else:
                        # Forward fill for categorical or other data
                        self.data[col] = self.data[col].ffill()
                        
            # Handle outliers using IQR method
            for col in self.data.select_dtypes(include=[np.number]).columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.warning(f"Found {outlier_count} outliers in {col}")
                    # Winsorize outliers
                    self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
                    
            logger.info("Data cleaning completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            return False
            
    def test_stationarity(self, series):
        """Test for stationarity using ADF and KPSS tests.
        
        Args:
            series (pd.Series): Time series to test
            
        Returns:
            dict: Dictionary containing test results
        """
        try:
            # ADF Test
            adf_result = adfuller(series.dropna())
            
            # KPSS Test
            kpss_result = kpss(series.dropna())
            
            results = {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'kpss_statistic': kpss_result[0],
                'kpss_pvalue': kpss_result[1],
                'is_stationary': adf_result[1] < 0.05 and kpss_result[1] >= 0.05
            }
            
            return results
        except Exception as e:
            logger.error(f"Error during stationarity testing: {e}")
            return None
            
    def create_features(self):
        """Create time series features for modeling."""
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return False
            
        try:
            # Add primary lags (focus on recent months)
            target = self.config['variables']['target_variable']
            for lag in [1, 2, 3]:  # Reduced from [1,2,3,6,12]
                self.data[f"{target}_lag_{lag}"] = self.data[target].shift(lag)
                
            # Add rolling statistics with more focused windows
            for window in [3, 6]:  # Removed 12-month window to reduce noise
                # Rolling mean
                self.data[f"{target}_roll_mean_{window}"] = \
                    self.data[target].rolling(window=window).mean()
                # Rolling std for volatility
                self.data[f"{target}_roll_std_{window}"] = \
                    self.data[target].rolling(window=window).std()
                
            # Calculate month-over-month changes
            self.data[f"{target}_mom_change"] = self.data[target].pct_change()
            self.data[f"{target}_mom_change_3m_avg"] = \
                self.data[f"{target}_mom_change"].rolling(window=3).mean()
                
            # Create interaction terms using appropriate transformations
            core_predictors = self.config['variables']['predictors']['core']
            for i, pred1 in enumerate(core_predictors):
                for pred2 in core_predictors[i+1:]:
                    if pred1 in self.data.columns and pred2 in self.data.columns:
                        # Use log versions for price interactions
                        col1 = f"{pred1}_log" if any(x in pred1.lower() for x in ['price', 'spot', 'futures']) else pred1
                        col2 = f"{pred2}_log" if any(x in pred2.lower() for x in ['price', 'spot', 'futures']) else pred2
                        
                        if col1 in self.data.columns and col2 in self.data.columns:
                            self.data[f"{pred1}_{pred2}_interact"] = \
                                self.data[col1] * self.data[col2]
                                
            logger.info("Feature creation completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during feature creation: {e}")
            return False
            
    def transform_data(self):
        """Apply necessary transformations to prepare data for modeling."""
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return False
            
        try:
            # Log transform price and rate variables
            target = self.config['variables']['target_variable']
            if (self.data[target] > 0).all():
                self.data[f"{target}_log"] = np.log(self.data[target])
                
            # Transform predictors based on their type
            for pred in self.config['variables']['predictors']['core']:
                if pred not in self.data.columns:
                    continue
                    
                # Log transform for price/rate variables
                if any(x in pred.lower() for x in ['price', 'spot', 'futures']):
                    if (self.data[pred] > 0).all():
                        self.data[f"{pred}_log"] = np.log(self.data[pred])
                else:
                    # Scale other numeric features
                    self.data[f"{pred}_scaled"] = stats.zscore(self.data[pred])
                    
            logger.info("Data transformation completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            return False
            
    def save_processed_data(self, filepath):
        """Save the processed data to a CSV file.
        
        Args:
            filepath (str): Path where to save the processed data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.data.to_csv(filepath)
            logger.info(f"Successfully saved processed data to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving processed data to {filepath}: {e}")
            return False

    def prepare_data(self, data):
        """Prepare data for modeling by selecting relevant features."""
        if data is None:
            logger.error("No data provided to prepare")
            return None, None
            
        try:
            # Get target variable (use log-transformed version)
            target_col = self.config['variables']['target_variable']
            target_log = f"{target_col}_log"
            y = data[target_log] if target_log in data.columns else data[target_col]
            
            # Initialize feature lists by category
            price_features = []
            technical_features = []
            macro_features = []
            seasonal_features = []
            
            # 1. Add transformed external predictors
            for pred in self.config['variables']['predictors']['core']:
                if pred in data.columns:
                    if any(x in pred.lower() for x in ['price', 'spot', 'futures']):
                        feat = f"{pred}_log"
                        if feat in data.columns:
                            price_features.append(feat)
                    else:
                        feat = f"{pred}_scaled"
                        if feat in data.columns:
                            macro_features.append(feat)
            
            # 2. Add technical indicators (time series features)
            # Recent lags (handle missing values with forward fill)
            for lag in [1, 2, 3]:
                col = f"{target_col}_lag_{lag}"
                if col in data.columns:
                    data[col] = data[col].ffill()
                    technical_features.append(col)
            
            # Rolling statistics (handle missing values with forward fill)
            for window in [3, 6]:
                mean_col = f"{target_col}_roll_mean_{window}"
                if mean_col in data.columns:
                    data[mean_col] = data[mean_col].ffill()
                    technical_features.append(mean_col)
                
                std_col = f"{target_col}_roll_std_{window}"
                if std_col in data.columns:
                    data[std_col] = data[std_col].ffill()
                    technical_features.append(std_col)
            
            # 3. Add seasonal components
            # Month indicators (1-12)
            data['month'] = data.index.month
            for month in range(1, 13):
                data[f'month_{month}'] = (data['month'] == month).astype(int)
                seasonal_features.append(f'month_{month}')
            
            # Quarter indicators (1-4)
            data['quarter'] = data.index.quarter
            for quarter in range(1, 5):
                data[f'quarter_{quarter}'] = (data['quarter'] == quarter).astype(int)
                seasonal_features.append(f'quarter_{quarter}')
            
            # 4. Add trend component
            data['time_index'] = np.arange(len(data))
            data['time_index_squared'] = data['time_index'] ** 2
            technical_features.extend(['time_index', 'time_index_squared'])
            
            # Combine all features
            selected_features = price_features + technical_features + macro_features + seasonal_features
            
            # Select features and handle any remaining missing values
            X = data[selected_features].copy()
            
            # Forward fill any remaining missing values
            X = X.ffill().bfill()  # Use backfill as last resort
            
            # Align X and y to handle any rows dropped due to NaN
            X, y = X.align(y, join='inner', axis=0)
            
            logger.info(f"Prepared data with {X.shape[1]} features:")
            logger.info(f"Price features: {len(price_features)}")
            logger.info(f"Technical features: {len(technical_features)}")
            logger.info(f"Macro features: {len(macro_features)}")
            logger.info(f"Seasonal features: {len(seasonal_features)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None

    def analyze_predictions(self, y_true_log, y_pred_log, target_col):
        """Analyze model predictions on both log and original scales.
        
        Args:
            y_true_log (pd.Series): True target values in log scale
            y_pred_log (pd.Series): Predicted values in log scale
            target_col (str): Name of the target variable
            
        Returns:
            dict: Dictionary containing analysis results
        """
        try:
            import matplotlib.pyplot as plt
            from scipy import stats
            
            # Convert predictions back to original scale
            y_true_orig = np.exp(y_true_log)
            y_pred_orig = np.exp(y_pred_log)
            
            # Calculate residuals
            residuals_log = y_true_log - y_pred_log
            residuals_orig = y_true_orig - y_pred_orig
            
            # Calculate metrics on both scales
            metrics = {
                'log_scale': {
                    'rmse': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
                    'mae': mean_absolute_error(y_true_log, y_pred_log),
                    'r2': r2_score(y_true_log, y_pred_log),
                    'mape': np.mean(np.abs((y_true_log - y_pred_log) / y_true_log)) * 100
                },
                'original_scale': {
                    'rmse': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
                    'mae': mean_absolute_error(y_true_orig, y_pred_orig),
                    'r2': r2_score(y_true_orig, y_pred_orig),
                    'mape': np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
                }
            }
            
            # Calculate naive forecast (previous value)
            naive_pred_log = y_true_log.shift(1)
            naive_pred_orig = y_true_orig.shift(1)
            
            # Calculate naive metrics (excluding first point due to shift)
            metrics['naive_log'] = {
                'rmse': np.sqrt(mean_squared_error(y_true_log[1:], naive_pred_log[1:])),
                'mae': mean_absolute_error(y_true_log[1:], naive_pred_log[1:]),
                'r2': r2_score(y_true_log[1:], naive_pred_log[1:]),
                'mape': np.mean(np.abs((y_true_log[1:] - naive_pred_log[1:]) / y_true_log[1:])) * 100
            }
            metrics['naive_orig'] = {
                'rmse': np.sqrt(mean_squared_error(y_true_orig[1:], naive_pred_orig[1:])),
                'mae': mean_absolute_error(y_true_orig[1:], naive_pred_orig[1:]),
                'r2': r2_score(y_true_orig[1:], naive_pred_orig[1:]),
                'mape': np.mean(np.abs((y_true_orig[1:] - naive_pred_orig[1:]) / y_true_orig[1:])) * 100
            }
            
            # Create diagnostic plots
            fig, axes = plt.subplots(3, 2, figsize=(15, 20))
            
            # Plot 1: Log Scale Time Series
            axes[0,0].plot(y_true_log.index, y_true_log.values, 'b-', label='Actual')
            axes[0,0].plot(y_pred_log.index, y_pred_log.values, 'r--', label='Predicted')
            axes[0,0].set_title(f'{target_col} (Log Scale)')
            axes[0,0].legend()
            axes[0,0].grid(True)
            
            # Plot 2: Original Scale Time Series
            axes[0,1].plot(y_true_orig.index, y_true_orig.values, 'b-', label='Actual')
            axes[0,1].plot(y_pred_orig.index, y_pred_orig.values, 'r--', label='Predicted')
            axes[0,1].set_title(f'{target_col} (Original Scale)')
            axes[0,1].legend()
            axes[0,1].grid(True)
            
            # Plot 3: Residuals over time
            axes[1,0].plot(residuals_log.index, residuals_log.values, 'b-', label='Log Scale')
            axes[1,0].axhline(y=0, color='r', linestyle='--')
            axes[1,0].set_title('Residuals Over Time')
            axes[1,0].legend()
            axes[1,0].grid(True)
            
            # Plot 4: Residuals vs Fitted
            axes[1,1].scatter(y_pred_log, residuals_log, alpha=0.5)
            axes[1,1].axhline(y=0, color='r', linestyle='--')
            axes[1,1].set_title('Residuals vs Fitted Values')
            axes[1,1].set_xlabel('Fitted Values (Log Scale)')
            axes[1,1].set_ylabel('Residuals')
            axes[1,1].grid(True)
            
            # Plot 5: Residuals Histogram
            axes[2,0].hist(residuals_log, bins=20, density=True, alpha=0.6)
            mu, sigma = stats.norm.fit(residuals_log)
            x = np.linspace(min(residuals_log), max(residuals_log), 100)
            p = stats.norm.pdf(x, mu, sigma)
            axes[2,0].plot(x, p, 'k', linewidth=2)
            axes[2,0].set_title('Residuals Distribution')
            axes[2,0].grid(True)
            
            # Plot 6: Q-Q Plot
            stats.probplot(residuals_log, dist="norm", plot=axes[2,1])
            axes[2,1].set_title('Q-Q Plot of Residuals')
            
            plt.tight_layout()
            
            # Save diagnostic plots
            plots_dir = os.path.join(PROJECT_ROOT, 'reports', 'figures')
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, 'model_diagnostics.png'))
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame.from_dict({
                (scale, metric): value
                for scale, metrics_dict in metrics.items()
                for metric, value in metrics_dict.items()
            }, orient='index', columns=['Value'])
            
            metrics_df.to_csv(os.path.join(plots_dir, 'model_metrics.csv'))
            
            # Calculate and print R² decomposition
            y_mean = y_true_log.mean()
            total_ss = np.sum((y_true_log - y_mean) ** 2)
            residual_ss = np.sum((y_true_log - y_pred_log) ** 2)
            r2_decomp = {
                'total_variance': total_ss,
                'residual_variance': residual_ss,
                'explained_variance': total_ss - residual_ss,
                'explained_variance_ratio': 1 - (residual_ss / total_ss)
            }
            
            metrics['r2_decomposition'] = r2_decomp
            
            logger.info("\nModel Performance Analysis:")
            logger.info("\nLog Scale Metrics:")
            for metric, value in metrics['log_scale'].items():
                logger.info(f"{metric}: {value:.4f}")
                
            logger.info("\nOriginal Scale Metrics:")
            for metric, value in metrics['original_scale'].items():
                logger.info(f"{metric}: {value:.4f}")
                
            logger.info("\nNaive Forecast Metrics (Log Scale):")
            for metric, value in metrics['naive_log'].items():
                logger.info(f"{metric}: {value:.4f}")
                
            logger.info("\nR² Decomposition:")
            for metric, value in r2_decomp.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during prediction analysis: {e}")
            return None

if __name__ == '__main__':
    from src.utils.logging_config import load_config, CONFIG_PATH
    
    # Load configuration
    config = load_config(CONFIG_PATH)
    if config is None:
        logger.critical("Failed to load configuration. Exiting.")
        sys.exit(1)
        
    # Example usage
    processor = DataProcessor(config)
    
    # Test with sample data if available
    raw_data_path = os.path.join(PROJECT_ROOT, config['data_paths']['raw'])
    processed_data_path = os.path.join(PROJECT_ROOT, config['data_paths']['processed'])
    
    # Example processing pipeline
    if os.path.exists(raw_data_path):
        for file in os.listdir(raw_data_path):
            if file.endswith('.csv'):
                logger.info(f"Processing {file}")
                if processor.load_data(os.path.join(raw_data_path, file)):
                    processor.clean_data()
                    processor.create_features()
                    processor.transform_data()
                    
                    # Save processed data
                    output_file = os.path.join(processed_data_path, f"processed_{file}")
                    processor.save_processed_data(output_file)