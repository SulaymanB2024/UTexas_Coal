"""Utilities for scenario generation and Monte Carlo simulation."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def generate_shock_path(
    base_value: float,
    shock_pct: float,
    duration: int,
    shock_type: str,
    forecast_horizon: int
) -> np.ndarray:
    """Generate a shock path for a variable based on shock parameters.
    
    Args:
        base_value: Starting value for the variable
        shock_pct: Percentage shock to apply (e.g., 0.4 for 40% increase)
        duration: Duration of shock in months
        shock_type: Either 'sustained' or 'temporary'
        forecast_horizon: Total number of months to generate
        
    Returns:
        Array of values representing the shocked path
    """
    path = np.zeros(forecast_horizon)
    shocked_value = base_value * (1 + shock_pct)
    
    if shock_type == 'sustained':
        path[:duration] = shocked_value
        path[duration:] = shocked_value
    elif shock_type == 'temporary':
        path[:duration] = shocked_value
        path[duration:] = base_value
    else:
        raise ValueError(f"Unknown shock type: {shock_type}")
        
    return path

def generate_scenario_paths(
    data: pd.DataFrame,
    scenario_config: Dict,
    forecast_horizon: int
) -> pd.DataFrame:
    """Generate exogenous variable paths for a given scenario.
    
    Args:
        data: Historical data with exogenous variables
        scenario_config: Configuration for the scenario
        forecast_horizon: Number of months to forecast
        
    Returns:
        DataFrame with paths for all exogenous variables
    """
    try:
        # Get the last values for all variables
        last_values = data.iloc[-1].copy()
        future_dates = pd.date_range(
            start=data.index[-1] + pd.DateOffset(months=1),
            periods=forecast_horizon,
            freq='M'
        )
        
        # Initialize with baseline persistence
        future_data = pd.DataFrame(
            index=future_dates,
            columns=data.columns,
            data=np.tile(last_values.values, (forecast_horizon, 1))
        )
        
        # If this is not the baseline scenario, apply shocks
        if scenario_config.get('type') != 'baseline':
            var_name = scenario_config['variable']
            if var_name not in future_data.columns:
                raise ValueError(f"Variable {var_name} not found in data")
                
            shocked_path = generate_shock_path(
                base_value=last_values[var_name],
                shock_pct=scenario_config['shock_pct'],
                duration=scenario_config['duration'],
                shock_type=scenario_config['type'],
                forecast_horizon=forecast_horizon
            )
            future_data[var_name] = shocked_path
            
        logger.info(f"Generated paths for scenario: {scenario_config['name']}")
        return future_data
        
    except Exception as e:
        logger.error(f"Error generating scenario paths: {e}")
        raise

def generate_error_paths(
    residuals: np.ndarray,
    num_sims: int,
    horizon: int,
    method: str = 'parametric',
    random_state: Optional[int] = None
) -> np.ndarray:
    """Generate paths of error terms for Monte Carlo simulation.
    
    Args:
        residuals: Historical model residuals
        num_sims: Number of Monte Carlo simulations
        horizon: Forecast horizon
        method: Either 'parametric' (normal) or 'bootstrap'
        random_state: Random seed for reproducibility
        
    Returns:
        Array of shape (num_sims, horizon) containing error paths
    """
    rng = np.random.RandomState(random_state)
    
    if method == 'parametric':
        # Generate from normal distribution
        std_dev = np.std(residuals)
        errors = rng.normal(0, std_dev, size=(num_sims, horizon))
    elif method == 'bootstrap':
        # Resample from historical residuals
        error_indices = rng.randint(0, len(residuals), size=(num_sims, horizon))
        errors = residuals[error_indices]
    else:
        raise ValueError(f"Unknown error generation method: {method}")
        
    return errors

def calculate_path_statistics(
    paths: np.ndarray,
    percentiles: List[float]
) -> pd.DataFrame:
    """Calculate summary statistics for simulated paths.
    
    Args:
        paths: Array of shape (num_sims, horizon) with simulated paths
        percentiles: List of percentiles to calculate
        
    Returns:
        DataFrame with summary statistics for each time step
    """
    stats = []
    horizon = paths.shape[1]
    
    for t in range(horizon):
        step_stats = {
            'horizon': t + 1,
            'mean': np.mean(paths[:, t]),
            'std': np.std(paths[:, t])
        }
        # Add percentiles
        for p in percentiles:
            step_stats[f'p{p}'] = np.percentile(paths[:, t], p)
            
        stats.append(step_stats)
        
    return pd.DataFrame(stats)