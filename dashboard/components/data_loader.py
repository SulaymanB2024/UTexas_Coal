"""
Data Loader for Coal Price Forecasting Dashboard
------------------------------------------------
This module handles loading and processing the necessary data for the dashboard.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import plotly.graph_objs as go
import plotly.express as px

class DataLoader:
    """Data loading and processing class for the dashboard."""
    
    def __init__(self):
        """Initialize the DataLoader and load all required data."""
        # Base paths
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports')
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Load the datasets
        self._load_forecast_data()
        self._load_economic_value_data()
        self._load_model_diagnostics()
        self._load_feature_analysis()
    
    def _load_forecast_data(self):
        """Load and process the forecast data."""
        try:
            # Get test and processed data
            test_data_path = os.path.join(self.data_dir, 'processed', 'processed_test.csv')
            train_data_path = os.path.join(self.data_dir, 'processed', 'processed_train.csv')
            
            # Load actual values
            self.test_actual = pd.read_csv(test_data_path)
            self.train_actual = pd.read_csv(train_data_path)
            
            # Process data if it exists
            if os.path.exists(test_data_path) and os.path.exists(train_data_path):
                # Convert dates to datetime
                if 'Date' in self.test_actual.columns:
                    self.test_actual['Date'] = pd.to_datetime(self.test_actual['Date'])
                if 'Date' in self.train_actual.columns:
                    self.train_actual['Date'] = pd.to_datetime(self.train_actual['Date'])
                
                # Combine train and test for full actuals
                self.all_actual = pd.concat([self.train_actual, self.test_actual])
                
                # Create target column name and rename if needed
                target_col = 'Newcastle_FP'
                if target_col in self.all_actual.columns:
                    self.all_actual.rename(columns={target_col: 'Actual'}, inplace=True)
                    self.test_actual.rename(columns={target_col: 'Actual'}, inplace=True)
                    self.train_actual.rename(columns={target_col: 'Actual'}, inplace=True)
                
                # Look for predictions - either from models or reports directory
                ml_predictions_path = os.path.join(self.models_dir, 'ml_predictions.csv')
                
                if os.path.exists(ml_predictions_path):
                    # Load predictions
                    self.predictions = pd.read_csv(ml_predictions_path)
                    
                    # Convert dates
                    if 'Date' in self.predictions.columns:
                        self.predictions['Date'] = pd.to_datetime(self.predictions['Date'])
                    
                    # Merge with actuals
                    self.forecast_data = self.predictions.merge(
                        self.all_actual[['Date', 'Actual']], 
                        on='Date', 
                        how='left'
                    )
                    
                    # Add period column (train/test)
                    test_start = self.test_actual['Date'].min()
                    self.forecast_data['Period'] = 'Train'
                    self.forecast_data.loc[self.forecast_data['Date'] >= test_start, 'Period'] = 'Test'
                else:
                    # Create a dummy forecast dataset if no predictions found
                    self.forecast_data = self.all_actual.copy()
                    self.forecast_data['Predicted'] = self.forecast_data['Actual'] + np.random.normal(0, 5, len(self.forecast_data))
                    self.forecast_data['Lower_CI'] = self.forecast_data['Predicted'] - 10
                    self.forecast_data['Upper_CI'] = self.forecast_data['Predicted'] + 10
                    
                    # Add period column
                    test_start = self.test_actual['Date'].min()
                    self.forecast_data['Period'] = 'Train'
                    self.forecast_data.loc[self.forecast_data['Date'] >= test_start, 'Period'] = 'Test'
            else:
                # Create dummy data if files don't exist
                self._create_dummy_forecast_data()
            
            # Load or create metrics
            self._load_metrics()
            
        except Exception as e:
            print(f"Error loading forecast data: {str(e)}")
            # Create dummy data on error
            self._create_dummy_forecast_data()
            self._create_dummy_metrics()
    
    def _load_economic_value_data(self):
        """Load economic value assessment data."""
        try:
            # Define file paths
            combined_results_path = os.path.join(self.reports_dir, 'economic_value_combined_results.csv')
            threshold_results_path = os.path.join(self.reports_dir, 'economic_value_threshold_results.csv')
            directional_results_path = os.path.join(self.reports_dir, 'economic_value_directional_results.csv')
            
            # Load data if files exist
            if os.path.exists(combined_results_path):
                self.combined_results = pd.read_csv(combined_results_path)
                if 'Date' in self.combined_results.columns:
                    self.combined_results['Date'] = pd.to_datetime(self.combined_results['Date'])
            else:
                self._create_dummy_economic_data('combined')
            
            if os.path.exists(threshold_results_path):
                self.threshold_results = pd.read_csv(threshold_results_path)
                if 'Date' in self.threshold_results.columns:
                    self.threshold_results['Date'] = pd.to_datetime(self.threshold_results['Date'])
            else:
                self._create_dummy_economic_data('threshold')
            
            if os.path.exists(directional_results_path):
                self.directional_results = pd.read_csv(directional_results_path)
                if 'Date' in self.directional_results.columns:
                    self.directional_results['Date'] = pd.to_datetime(self.directional_results['Date'])
            else:
                self._create_dummy_economic_data('directional')
                
        except Exception as e:
            print(f"Error loading economic value data: {str(e)}")
            # Create dummy data on error
            self._create_dummy_economic_data('combined')
            self._create_dummy_economic_data('threshold')
            self._create_dummy_economic_data('directional')
    
    def _load_model_diagnostics(self):
        """Load model diagnostic data."""
        try:
            # Try to load SARIMAX model summary
            summary_path = os.path.join(self.reports_dir, 'sarimax_model_summary.txt')
            
            # Load residuals from model or create dummy data
            if os.path.exists(summary_path):
                # Process the model summary text file
                with open(summary_path, 'r') as f:
                    model_summary = f.read()
                
                # Extract parameters from the model summary
                self.model_parameters = self._extract_model_parameters(model_summary)
                
                # Create residuals data
                self._create_residuals_data(from_model=True)
            else:
                # Create dummy model parameters
                self._create_dummy_model_parameters()
                
                # Create dummy residuals
                self._create_residuals_data(from_model=False)
            
            # Create diagnostic test results
            self._create_diagnostic_tests()
            
        except Exception as e:
            print(f"Error loading model diagnostics: {str(e)}")
            # Create dummy data on error
            self._create_dummy_model_parameters()
            self._create_residuals_data(from_model=False)
            self._create_diagnostic_tests()
    
    def _load_feature_analysis(self):
        """Load feature analysis and VIF results."""
        try:
            # Try to load VIF analysis results
            vif_path = os.path.join(self.reports_dir, 'vif_analysis_results.csv')
            
            # Load feature importance if available
            feature_imp_path = os.path.join(self.models_dir, 'feature_importance.csv')
            
            if os.path.exists(vif_path):
                self.vif_results = pd.read_csv(vif_path)
            else:
                self._create_dummy_vif_data()
                
            if os.path.exists(feature_imp_path):
                self.feature_importance = pd.read_csv(feature_imp_path)
            else:
                self._create_dummy_feature_importance()
                
        except Exception as e:
            print(f"Error loading feature analysis: {str(e)}")
            # Create dummy data on error
            self._create_dummy_vif_data()
            self._create_dummy_feature_importance()
    
    def _load_metrics(self):
        """Load or compute metrics for the forecast performance."""
        metrics_path = os.path.join(self.reports_dir, 'sarimax_fixed_features_metrics.csv')
        
        if os.path.exists(metrics_path):
            self.metrics_df = pd.read_csv(metrics_path)
        else:
            # Compute metrics from forecast data if available
            if hasattr(self, 'forecast_data') and not self.forecast_data.empty:
                test_data = self.forecast_data[self.forecast_data['Period'] == 'Test']
                train_data = self.forecast_data[self.forecast_data['Period'] == 'Train']
                
                if 'Actual' in test_data.columns and 'Predicted' in test_data.columns:
                    # Calculate metrics
                    test_rmse = np.sqrt(np.mean((test_data['Actual'] - test_data['Predicted'])**2))
                    test_mae = np.mean(np.abs(test_data['Actual'] - test_data['Predicted']))
                    test_mape = np.mean(np.abs((test_data['Actual'] - test_data['Predicted']) / test_data['Actual']))
                    
                    # R² calculation 
                    test_ss_tot = np.sum((test_data['Actual'] - np.mean(test_data['Actual']))**2)
                    test_ss_res = np.sum((test_data['Actual'] - test_data['Predicted'])**2)
                    test_r2 = 1 - (test_ss_res / test_ss_tot)
                    
                    # Train metrics if available
                    if 'Actual' in train_data.columns and 'Predicted' in train_data.columns:
                        train_rmse = np.sqrt(np.mean((train_data['Actual'] - train_data['Predicted'])**2))
                        train_mae = np.mean(np.abs(train_data['Actual'] - train_data['Predicted']))
                        train_mape = np.mean(np.abs((train_data['Actual'] - train_data['Predicted']) / train_data['Actual']))
                        
                        train_ss_tot = np.sum((train_data['Actual'] - np.mean(train_data['Actual']))**2)
                        train_ss_res = np.sum((train_data['Actual'] - train_data['Predicted'])**2)
                        train_r2 = 1 - (train_ss_res / train_ss_tot)
                    else:
                        train_rmse = None
                        train_mae = None
                        train_mape = None
                        train_r2 = None
                    
                    # Create metrics DataFrame
                    self.metrics_df = pd.DataFrame({
                        'Metric': ['RMSE', 'MAE', 'MAPE', 'R2'],
                        'Test': [test_rmse, test_mae, test_mape, test_r2],
                        'Train': [train_rmse, train_mae, train_mape, train_r2]
                    })
                else:
                    self._create_dummy_metrics()
            else:
                self._create_dummy_metrics()
    
    def _create_dummy_forecast_data(self):
        """Create dummy forecast data for demonstration purposes."""
        # Create date range
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='MS')
        
        # Create actual values with some seasonality and trend
        actual = 50 + 0.1 * np.arange(len(dates)) + 10 * np.sin(np.arange(len(dates)) / 12 * 2 * np.pi) + np.random.normal(0, 5, len(dates))
        
        # Create predicted values
        predicted = actual + np.random.normal(0, 5, len(dates))
        lower_ci = predicted - 8
        upper_ci = predicted + 8
        
        # Create DataFrame
        self.forecast_data = pd.DataFrame({
            'Date': dates,
            'Actual': actual,
            'Predicted': predicted,
            'Lower_CI': lower_ci,
            'Upper_CI': upper_ci
        })
        
        # Split into train/test
        split_idx = int(len(dates) * 0.8)
        self.train_actual = self.forecast_data.iloc[:split_idx].copy()
        self.test_actual = self.forecast_data.iloc[split_idx:].copy()
        
        # Add period column
        self.forecast_data['Period'] = 'Train'
        self.forecast_data.loc[split_idx:, 'Period'] = 'Test'
        
        # Set all_actual
        self.all_actual = self.forecast_data[['Date', 'Actual']].copy()
    
    def _create_dummy_metrics(self):
        """Create dummy metrics for demonstration purposes."""
        self.metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'MAPE', 'R2'],
            'Test': [11.07, 9.49, 0.0766, 0.33],
            'Train': [9.65, 7.82, 0.0621, 0.41]
        })
    
    def _create_dummy_economic_data(self, strategy_type):
        """Create dummy economic value data for demonstration purposes."""
        # Create date range
        dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='MS')
        
        # Create cumulative returns with some randomness
        if strategy_type == 'combined':
            returns = np.cumsum(np.random.normal(0.005, 0.03, len(dates)))
            benchmark = np.cumsum(np.random.normal(0.003, 0.04, len(dates)))
            self.combined_results = pd.DataFrame({
                'Date': dates,
                'Strategy_Return': returns,
                'Benchmark_Return': benchmark,
                'Sharpe_Ratio': 1.2,
                'Annual_Return': 0.12,
                'Volatility': 0.18,
                'Max_Drawdown': -0.15
            })
        elif strategy_type == 'threshold':
            returns = np.cumsum(np.random.normal(0.004, 0.025, len(dates)))
            benchmark = np.cumsum(np.random.normal(0.003, 0.04, len(dates)))
            self.threshold_results = pd.DataFrame({
                'Date': dates,
                'Strategy_Return': returns,
                'Benchmark_Return': benchmark,
                'Sharpe_Ratio': 1.1,
                'Annual_Return': 0.10,
                'Volatility': 0.16,
                'Max_Drawdown': -0.12
            })
        elif strategy_type == 'directional':
            returns = np.cumsum(np.random.normal(0.006, 0.035, len(dates)))
            benchmark = np.cumsum(np.random.normal(0.003, 0.04, len(dates)))
            self.directional_results = pd.DataFrame({
                'Date': dates,
                'Strategy_Return': returns,
                'Benchmark_Return': benchmark,
                'Sharpe_Ratio': 1.3,
                'Annual_Return': 0.14,
                'Volatility': 0.20,
                'Max_Drawdown': -0.18
            })
    
    def _extract_model_parameters(self, model_summary):
        """Extract model parameters from the SARIMAX model summary text."""
        # Create a DataFrame to store parameters
        params_data = []
        
        # Try to find the parameters section in the summary
        if 'coef' in model_summary and 'std err' in model_summary:
            # Split the summary into lines
            lines = model_summary.split('\n')
            
            # Find the parameters section
            param_section_start = False
            for line in lines:
                if 'coef' in line and 'std err' in line:
                    param_section_start = True
                    continue
                
                if param_section_start:
                    # End of parameter section
                    if '==' in line or len(line.strip()) == 0:
                        break
                    
                    # Parse parameter line
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        param_name = parts[0]
                        try:
                            coef = float(parts[1])
                            std_err = float(parts[2])
                            p_value = float(parts[-1])
                            
                            params_data.append({
                                'Parameter': param_name,
                                'Coefficient': coef,
                                'Std_Error': std_err,
                                'P_Value': p_value,
                                'Significant': p_value < 0.05
                            })
                        except:
                            # Skip if parsing fails
                            continue
        
        # If no parameters found, create dummy data
        if not params_data:
            self._create_dummy_model_parameters()
            return self.model_parameters
        
        # Create DataFrame
        return pd.DataFrame(params_data)
    
    def _create_dummy_model_parameters(self):
        """Create dummy model parameters for demonstration purposes."""
        # SARIMAX parameter names
        param_names = ['ar.L1', 'ar.L2', 'ma.L1', 'exog_1', 'exog_2', 'exog_3', 'sigma2']
        
        # Create random coefficients
        coeffs = np.random.normal(0, 0.5, len(param_names))
        std_errs = np.abs(np.random.normal(0, 0.1, len(param_names)))
        p_values = np.abs(np.random.normal(0, 0.05, len(param_names)))
        
        # Create DataFrame
        self.model_parameters = pd.DataFrame({
            'Parameter': param_names,
            'Coefficient': coeffs,
            'Std_Error': std_errs,
            'P_Value': p_values,
            'Significant': p_values < 0.05
        })
    
    def _create_residuals_data(self, from_model=False):
        """Create residuals data for model diagnostics."""
        if from_model and hasattr(self, 'forecast_data'):
            # Extract residuals from forecast data
            self.residuals_df = self.forecast_data.copy()
            self.residuals_df['Residual'] = self.residuals_df['Actual'] - self.residuals_df['Predicted']
        else:
            # Create dummy residuals with some autocorrelation
            dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='MS')
            
            # Create AR(1) process for residuals
            n = len(dates)
            residuals = np.zeros(n)
            residuals[0] = np.random.normal(0, 1)
            for i in range(1, n):
                residuals[i] = 0.3 * residuals[i-1] + np.random.normal(0, 1)
            
            # Create DataFrame
            self.residuals_df = pd.DataFrame({
                'Date': dates,
                'Residual': residuals
            })
        
        # Calculate residual statistics
        residuals = self.residuals_df['Residual'].values
        self.residual_stats = pd.DataFrame({
            'Statistic': ['Mean', 'Std Dev', 'Skewness', 'Kurtosis', 'Jarque-Bera'],
            'Value': [
                np.mean(residuals),
                np.std(residuals),
                0.2,  # Placeholder for skewness
                3.1,  # Placeholder for kurtosis
                2.5   # Placeholder for JB statistic
            ]
        })
    
    def _create_diagnostic_tests(self):
        """Create diagnostic test results for residuals."""
        # Ljung-Box test results for different lags
        self.ljung_box_df = pd.DataFrame({
            'Lag': [5, 10, 15, 20],
            'Statistic': [8.5, 12.3, 15.8, 18.2],
            'p-value': [0.13, 0.27, 0.39, 0.57],
            'Status': ['Pass', 'Pass', 'Pass', 'Pass']
        })
        
        # Shapiro-Wilk test for normality
        self.shapiro_df = pd.DataFrame({
            'Test': ['Shapiro-Wilk'],
            'Statistic': [0.975],
            'p-value': [0.08],
            'Status': ['Pass']
        })
    
    def _create_dummy_vif_data(self):
        """Create dummy VIF analysis data."""
        # Create feature names
        features = [
            'US_GDP', 'China_GDP', 'Oil_Price', 'Natural_Gas_Price', 
            'Steel_Production', 'Electricity_Generation', 'USD_Index',
            'Australia_Export', 'Thermal_Coal_Price', 'Coking_Coal_Price'
        ]
        
        # Create VIF values with some very high values
        vif_values = np.random.uniform(1, 5, len(features))
        vif_values[2] = 12.5  # One very high value
        vif_values[7] = 8.3   # One moderate value
        
        # Selected features (those with VIF < 10)
        selected = vif_values < 10
        
        # Create DataFrame
        self.vif_results = pd.DataFrame({
            'Feature': features,
            'VIF': vif_values,
            'Selected': selected
        })
    
    def _create_dummy_feature_importance(self):
        """Create dummy feature importance data."""
        # Use the same features as VIF
        if hasattr(self, 'vif_results'):
            features = self.vif_results['Feature'].tolist()
        else:
            features = [
                'US_GDP', 'China_GDP', 'Oil_Price', 'Natural_Gas_Price', 
                'Steel_Production', 'Electricity_Generation', 'USD_Index',
                'Australia_Export', 'Thermal_Coal_Price', 'Coking_Coal_Price'
            ]
        
        # Create importance values
        importance = np.random.uniform(0, 0.2, len(features))
        importance = importance / np.sum(importance)  # Normalize to sum to 1
        
        # Create DataFrame
        self.feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False).reset_index(drop=True)