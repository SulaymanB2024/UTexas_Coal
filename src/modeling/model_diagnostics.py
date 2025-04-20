"""Model diagnostics and enhanced visualization utilities."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import logging
from typing import Dict, List, Optional, Tuple, Any
import os

logger = logging.getLogger(__name__)

def save_sample_paths(
    results: Dict[str, Dict],
    num_paths: int,
    output_file: str
) -> None:
    """Save a subset of Monte Carlo simulated paths for each scenario.
    
    Args:
        results: Dictionary containing scenario results
        num_paths: Number of paths to save per scenario
        output_file: Path to save the results
    """
    sample_paths = {}
    
    for scenario_id, result in results.items():
        paths = result['simulated_paths'][:num_paths, :]
        scenario_name = result['config']['name']
        
        # Create DataFrame for this scenario's paths
        for i in range(num_paths):
            col_name = f"{scenario_name}_path_{i+1}"
            sample_paths[col_name] = paths[i, :]
            
    # Save to CSV
    pd.DataFrame(sample_paths).to_csv(output_file)
    logger.info(f"Saved {num_paths} sample paths per scenario to {output_file}")

def plot_forecast_distributions(
    results: Dict[str, Dict],
    horizons: List[int],
    output_dir: str
) -> None:
    """Generate distribution plots for specific forecast horizons.
    
    Args:
        results: Dictionary containing scenario results
        horizons: List of forecast horizons to analyze
        output_dir: Directory to save plots
    """
    for horizon in horizons:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                results[s]['config']['name'] 
                for s in ['baseline', 'carbon_shock', 'gas_shock', 'freight_shock']
            ]
        )
        
        for i, (scenario_id, result) in enumerate(results.items()):
            # Get prices at this horizon for all simulations
            prices = result['simulated_paths'][:, horizon-1]
            
            # Create histogram
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Histogram(
                    x=prices,
                    name=result['config']['name'],
                    nbinsx=30,
                    histnorm='probability'
                ),
                row=row,
                col=col
            )
            
            # Add KDE
            kde_x = np.linspace(min(prices), max(prices), 100)
            kde = stats.gaussian_kde(prices)
            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde(kde_x),
                    name=f"{result['config']['name']} KDE",
                    line=dict(dash='dash')
                ),
                row=row,
                col=col
            )
            
        fig.update_layout(
            title=f"Price Distribution at {horizon}-Month Horizon",
            showlegend=True,
            height=800
        )
        
        output_file = os.path.join(
            output_dir,
            f"forecast_distribution_h{horizon}.html"
        )
        fig.write_html(output_file)
        logger.info(f"Saved forecast distribution plot for horizon {horizon} to {output_file}")

def plot_model_diagnostics(residuals, config=None):
    """Plot model diagnostics including ACF, PACF, and QQ plot."""
    # Calculate maximum lags dynamically based on sample size
    nobs = len(residuals)
    max_lags = 40  # Default maximum lags
    if config and 'max_lags' in config:
        max_lags = config['max_lags']
    
    # Calculate nlags to prevent ValueError with short series
    nlags_to_use = min(max_lags, nobs // 2 - 1)
    
    # Ensure residuals are numpy array and handle any NaN values
    residuals = np.array(residuals)
    residuals = residuals[~np.isnan(residuals)]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Residuals', 'Normal Q-Q', 'ACF', 'PACF')
    )
    
    # Residuals plot
    fig.add_trace(
        go.Scatter(y=residuals, mode='lines', name='Residuals'),
        row=1, col=1
    )
    
    # Q-Q plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sorted_residuals = np.sort(residuals)
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers',
                  name='Q-Q Plot'),
        row=1, col=2
    )
    
    # Add reference line for Q-Q plot
    min_val = min(theoretical_quantiles)
    max_val = max(theoretical_quantiles)
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', name='Reference Line',
                  line=dict(dash='dash')),
        row=1, col=2
    )
    
    # ACF plot with error handling
    try:
        acf_values = acf(residuals, nlags=nlags_to_use, fft=True)
        fig.add_trace(
            go.Scatter(y=acf_values, mode='lines+markers', name='ACF'),
            row=2, col=1
        )
    except Exception as e:
        logger.warning(f"Error calculating ACF: {str(e)}")
        acf_values = np.zeros(nlags_to_use + 1)
        fig.add_trace(
            go.Scatter(y=acf_values, mode='lines+markers', name='ACF (Error)'),
            row=2, col=1
        )
    
    # PACF plot with error handling and robust calculation
    try:
        # Use OLS method which is more stable for PACF calculation
        pacf_values = pacf(residuals, nlags=nlags_to_use, method='ols')
        fig.add_trace(
            go.Scatter(y=pacf_values, mode='lines+markers', name='PACF'),
            row=2, col=2
        )
    except Exception as e:
        # If OLS method fails, try alternative methods
        try:
            pacf_values = pacf(residuals, nlags=nlags_to_use, method='ywa')
            fig.add_trace(
                go.Scatter(y=pacf_values, mode='lines+markers', name='PACF'),
                row=2, col=2
            )
        except Exception as e2:
            logger.warning(f"Error calculating PACF: {str(e2)}")
            pacf_values = np.zeros(nlags_to_use + 1)
            fig.add_trace(
                go.Scatter(y=pacf_values, mode='lines+markers', name='PACF (Error)'),
                row=2, col=2
            )
    
    # Add confidence intervals
    confidence_interval = 1.96 / np.sqrt(len(residuals))
    for row in [2]:
        for col in [1, 2]:
            fig.add_hline(y=confidence_interval, line_dash="dash", 
                         line_color="red", annotation_text="95% CI", 
                         row=row, col=col)
            fig.add_hline(y=-confidence_interval, line_dash="dash", 
                         line_color="red", row=row, col=col)
    
    # Update layout
    fig.update_layout(height=800, width=1000, showlegend=False,
                     title_text="Model Diagnostic Plots")
    
    # Save the plot
    fig.write_html("reports/figures/model_diagnostics.html")
    
    return fig

def compare_in_out_sample_metrics(
    model: Any,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_var: str
) -> Dict[str, Dict[str, float]]:
    """Compare in-sample and out-of-sample model performance.
    
    Args:
        model: Fitted model object
        train_data: Training data
        test_data: Test data
        target_var: Name of target variable
        
    Returns:
        Dictionary containing metrics for both train and test sets
    """
    # Ensure proper datetime index
    train_data.index = pd.to_datetime(train_data.index)
    test_data.index = pd.to_datetime(test_data.index)
    
    # Generate predictions
    X_train = train_data.drop(columns=[target_var])
    y_train = train_data[target_var]
    train_pred, _ = model.predict(X_train)
    
    X_test = test_data.drop(columns=[target_var])
    y_test = test_data[target_var]
    test_pred, _ = model.predict(X_test)
    
    # Ensure predictions have proper datetime index
    train_pred.index = pd.to_datetime(train_pred.index)
    test_pred.index = pd.to_datetime(test_pred.index)
    
    # Align indices
    common_train_idx = train_pred.index.intersection(y_train.index)
    common_test_idx = test_pred.index.intersection(y_test.index)
    
    # Select data with aligned indices
    y_train_aligned = y_train[common_train_idx]
    train_pred_aligned = train_pred[common_train_idx]
    y_test_aligned = y_test[common_test_idx]
    test_pred_aligned = test_pred[common_test_idx]
    
    # Calculate metrics
    metrics = {
        'train': {},
        'test': {}
    }
    
    # Training metrics
    metrics['train']['mae'] = mean_absolute_error(y_train_aligned, train_pred_aligned)
    metrics['train']['rmse'] = np.sqrt(mean_squared_error(y_train_aligned, train_pred_aligned))
    metrics['train']['mape'] = mean_absolute_percentage_error(y_train_aligned, train_pred_aligned)
    metrics['train']['r2'] = r2_score(y_train_aligned, train_pred_aligned)
    
    # Test metrics
    metrics['test']['mae'] = mean_absolute_error(y_test_aligned, test_pred_aligned)
    metrics['test']['rmse'] = np.sqrt(mean_squared_error(y_test_aligned, test_pred_aligned))
    metrics['test']['mape'] = mean_absolute_percentage_error(y_test_aligned, test_pred_aligned)
    metrics['test']['r2'] = r2_score(y_test_aligned, test_pred_aligned)
    
    return metrics

def save_model_summary(model, filepath):
    """Save model summary to a text file.
    
    Args:
        model: Fitted model object
        filepath: Path to save the summary
    """
    with open(filepath, 'w') as f:
        # Handle different model types
        if hasattr(model, 'result'):
            summary = model.result.summary()
        elif hasattr(model, 'model') and hasattr(model.model, 'result'):
            summary = model.model.result.summary()
        else:
            summary = str(model.params) if hasattr(model, 'params') else str(model)
            
        f.write(str(summary))
        
        # Add diagnostic test results if available
        if hasattr(model, 'get_diagnostics'):
            f.write("\n\nDiagnostic Tests:\n")
            diagnostics = model.get_diagnostics()
            for test_name, results in diagnostics.items():
                if test_name not in ['residuals', 'aic', 'bic']:
                    f.write(f"\n{test_name}:")
                    f.write(f"\n\tStatistic: {results['statistic']}")
                    f.write(f"\n\tp-value: {results['pvalue']}")
            
            f.write(f"\n\nInformation Criteria:")
            f.write(f"\nAIC: {diagnostics.get('aic', 'N/A')}")
            f.write(f"\nBIC: {diagnostics.get('bic', 'N/A')}")