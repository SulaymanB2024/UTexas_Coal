"""Monte Carlo Scenario Analysis for Coal Price Forecasting."""
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils.logging_config import setup_logging, load_config, CONFIG_PATH
from src.modeling.ts_models import TSModels
from src.data_processing.scenario_utils import (
    generate_scenario_paths,
    generate_error_paths,
    calculate_path_statistics
)
from src.modeling.model_diagnostics import (
    save_sample_paths,
    plot_forecast_distributions,
    plot_model_diagnostics,
    compare_in_out_sample_metrics,
    save_model_summary
)

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

class ScenarioAnalysis:
    """Implements Monte Carlo scenario analysis using the fitted ARIMAX model."""
    
    def __init__(self, config: Dict, model: TSModels):
        """Initialize scenario analysis.
        
        Args:
            config: Project configuration dictionary
            model: Fitted ARIMAX model
        """
        self.config = config
        self.model = model
        self.scenario_config = config['scenario_analysis']
        self.results = {}
        
    def run_monte_carlo_simulation(
        self,
        scenario_name: str,
        historical_data: pd.DataFrame,
        future_exog: pd.DataFrame,
        residuals: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Run Monte Carlo simulation for a given scenario.
        
        Args:
            scenario_name: Name of the scenario
            historical_data: Historical data including target and features
            future_exog: Future paths of exogenous variables
            residuals: Historical model residuals for error sampling
            
        Returns:
            Tuple of (simulated paths array, summary statistics DataFrame)
        """
        num_sims = self.scenario_config['general']['num_simulations']
        horizon = self.scenario_config['general']['forecast_horizon']
        random_seed = self.scenario_config['general']['random_seed']
        
        # Generate error paths
        error_paths = generate_error_paths(
            residuals=residuals,
            num_sims=num_sims,
            horizon=horizon,
            method='parametric',
            random_state=random_seed
        )
        
        # Initialize array for simulated paths
        simulated_paths = np.zeros((num_sims, horizon))
        
        # Get last values for lagged terms
        last_target = historical_data[self.config['variables']['target_variable']].iloc[-1]
        
        logger.info(f"Starting Monte Carlo simulation for scenario: {scenario_name}")
        logger.info(f"Number of simulations: {num_sims}")
        
        # Run simulations
        for i in range(num_sims):
            if i % 100 == 0:
                logger.info(f"Completed {i} simulations...")
                
            current_path = []
            current_value = last_target
            
            # Generate each step of the path
            for t in range(horizon):
                # Prepare features for this step
                step_features = future_exog.iloc[t:t+1].copy()
                
                # Add the error term for this step
                forecast, _ = self.model.predict(step_features)
                simulated_value = forecast.iloc[0] + error_paths[i, t]
                
                current_path.append(simulated_value)
                current_value = simulated_value
                
            simulated_paths[i, :] = current_path
            
        # Calculate summary statistics
        stats_df = calculate_path_statistics(
            simulated_paths,
            self.scenario_config['output_settings']['percentiles']
        )
        
        logger.info(f"Completed Monte Carlo simulation for scenario: {scenario_name}")
        return simulated_paths, stats_df
        
    def run_all_scenarios(self, historical_data: pd.DataFrame) -> Dict:
        """Run Monte Carlo simulation for all configured scenarios.
        
        Args:
            historical_data: Historical data including target and features
            
        Returns:
            Dictionary containing results for each scenario
        """
        horizon = self.scenario_config['general']['forecast_horizon']
        
        # Get model residuals
        target_var = self.config['variables']['target_variable']
        predictions, _ = self.model.predict(historical_data.drop(columns=[target_var]))
        residuals = historical_data[target_var].values - predictions.values
        
        # Process each scenario
        for scenario_id, scenario_config in self.scenario_config['scenarios'].items():
            logger.info(f"\nProcessing scenario: {scenario_config['name']}")
            
            # Generate exogenous variable paths for this scenario
            future_exog = generate_scenario_paths(
                data=historical_data.drop(columns=[target_var]),
                scenario_config=scenario_config,
                forecast_horizon=horizon
            )
            
            # Run Monte Carlo simulation
            paths, stats = self.run_monte_carlo_simulation(
                scenario_name=scenario_config['name'],
                historical_data=historical_data,
                future_exog=future_exog,
                residuals=residuals
            )
            
            self.results[scenario_id] = {
                'config': scenario_config,
                'exog_paths': future_exog,
                'simulated_paths': paths,
                'statistics': stats
            }
            
        return self.results
        
    def save_results(self):
        """Save scenario analysis results to files."""
        if not self.results:
            logger.warning("No results to save. Run scenarios first.")
            return
            
        # Create consolidated statistics DataFrame
        stats_data = []
        for scenario_id, result in self.results.items():
            scenario_stats = result['statistics'].copy()
            scenario_stats['scenario'] = result['config']['name']
            stats_data.append(scenario_stats)
            
        all_stats = pd.concat(stats_data, axis=0, ignore_index=True)
        
        # Save to CSV
        output_file = os.path.join(
            PROJECT_ROOT,
            self.scenario_config['output_settings']['results_file']
        )
        all_stats.to_csv(output_file, index=False)
        logger.info(f"Saved scenario statistics to: {output_file}")
        
    def plot_results(self, historical_data: pd.DataFrame):
        """Create interactive plot of scenario analysis results."""
        if not self.results:
            logger.warning("No results to plot. Run scenarios first.")
            return
            
        target_var = self.config['variables']['target_variable']
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data[target_var],
            name='Historical',
            line=dict(color='black'),
            showlegend=True
        ))
        
        # Colors for scenarios
        colors = {
            'baseline': 'blue',
            'carbon_shock': 'red',
            'gas_shock': 'green',
            'freight_shock': 'purple'
        }
        
        # Add scenarios
        for scenario_id, result in self.results.items():
            stats = result['statistics']
            config = result['config']
            color = colors.get(scenario_id, 'gray')
            
            # Add median line
            fig.add_trace(go.Scatter(
                x=pd.date_range(
                    start=historical_data.index[-1],
                    periods=len(stats) + 1,
                    freq='M'
                ),
                y=np.concatenate([[historical_data[target_var].iloc[-1]], stats['p50']]),
                name=f"{config['name']} (Median)",
                line=dict(color=color)
            ))
            
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=pd.date_range(
                    start=historical_data.index[-1],
                    periods=len(stats),
                    freq='M'
                ),
                y=stats['p95'],
                name=f"{config['name']} (95th)",
                line=dict(color=color, dash='dash'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=pd.date_range(
                    start=historical_data.index[-1],
                    periods=len(stats),
                    freq='M'
                ),
                y=stats['p5'],
                name=f"{config['name']} (5th)",
                line=dict(color=color, dash='dash'),
                fill='tonexty',
                showlegend=False
            ))
            
        # Update layout
        fig.update_layout(
            title=f"Monte Carlo Scenario Analysis - {target_var}",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            template="plotly_white",
            showlegend=True
        )
        
        # Save plot
        output_file = os.path.join(
            PROJECT_ROOT,
            self.scenario_config['output_settings']['plot_file']
        )
        fig.write_html(output_file)
        logger.info(f"Saved scenario plot to: {output_file}")

def main():
    """Run the scenario analysis pipeline."""
    try:
        # Load configuration
        config = load_config(CONFIG_PATH)
        if config is None:
            raise ValueError("Failed to load configuration")
            
        # Load data
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
        
        # Load and fit the model
        model = TSModels(config)
        target_var = config['variables']['target_variable']
        X_train = train_data.drop(columns=[target_var])
        y_train = train_data[target_var]
        
        logger.info("Fitting ARIMAX model...")
        model.fit(X_train, y_train)
        
        # Save detailed model summary
        save_model_summary(
            model,
            os.path.join(PROJECT_ROOT, 'reports/arimax_model_summary.txt')
        )
        
        # Generate and save model diagnostics
        train_pred, _ = model.predict(X_train)
        residuals = y_train.values - train_pred.values
        
        diagnostic_results = plot_model_diagnostics(
            model,
            residuals,
            os.path.join(PROJECT_ROOT, 'reports/figures')
        )
        
        logger.info("Model Diagnostic Test Results:")
        logger.info(f"Shapiro-Wilk test p-value: {diagnostic_results['shapiro_wilk_pvalue']:.4f}")
        for lag, pval in diagnostic_results['ljung_box_pvalues'].items():
            logger.info(f"Ljung-Box test ({lag}): {pval:.4f}")
            
        # Compare in-sample vs out-of-sample performance
        metric_comparison = compare_in_out_sample_metrics(
            model,
            train_data,
            test_data,
            target_var
        )
        
        logger.info("\nPerformance Metric Comparison:")
        logger.info("Training Set Metrics:")
        for metric, value in metric_comparison['train'].items():
            logger.info(f"{metric}: {value:.4f}")
            
        logger.info("\nTest Set Metrics:")
        for metric, value in metric_comparison['test'].items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Run scenario analysis
        scenario_analysis = ScenarioAnalysis(config, model)
        results = scenario_analysis.run_all_scenarios(train_data)
        
        # Save scenario results
        scenario_analysis.save_results()
        scenario_analysis.plot_results(train_data)
        
        # Save sample paths for detailed analysis
        save_sample_paths(
            results,
            num_paths=50,
            output_file=os.path.join(PROJECT_ROOT, 'reports/scenario_analysis_mc_sample_paths.csv')
        )
        
        # Generate distribution plots for key horizons
        plot_forecast_distributions(
            results,
            horizons=[1, 3, 6, 12],
            output_dir=os.path.join(PROJECT_ROOT, 'reports/figures')
        )
        
        logger.info("Scenario analysis and diagnostics completed successfully")
        
    except Exception as e:
        logger.error(f"Error in scenario analysis: {e}")
        raise

if __name__ == "__main__":
    main()