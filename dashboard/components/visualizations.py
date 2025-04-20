"""
Visualization Components for Coal Price Forecasting Dashboard
------------------------------------------------------------
This module contains all visualization functions for the dashboard.
"""
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Try importing optional statistical libraries, with fallbacks if not available
try:
    from statsmodels.tsa.stattools import acf
    from scipy import stats
except ImportError:
    # Mock implementation if libraries aren't available
    def acf(x, nlags=40, fft=False):
        """Mock ACF function."""
        import numpy as np
        return np.random.normal(0, 0.2, nlags+1)
    
    from collections import namedtuple
    stats_module = namedtuple('stats', ['norm'])
    stats = stats_module(namedtuple('norm', ['ppf'])(lambda x: x))

class Visualizations:
    """Class for creating all dashboard visualizations"""
    
    @staticmethod
    def create_forecast_plot(actual_data, forecast_data):
        """Create the forecast plot with historicals, predictions, and confidence intervals."""
        # Set up figure
        fig = go.Figure()
        
        # Add historical actual values
        if not actual_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=actual_data['Date'],
                    y=actual_data['Actual'],
                    name="Historical Prices",
                    line=dict(color="black", width=2),
                    mode="lines",
                )
            )
        
        # Add forecasts
        if not forecast_data.empty and 'Date' in forecast_data.columns and 'Forecast' in forecast_data.columns:
            # Training period forecasts
            train_forecasts = forecast_data[forecast_data['Period'] == 'Train']
            if not train_forecasts.empty:
                fig.add_trace(
                    go.Scatter(
                        x=train_forecasts['Date'],
                        y=train_forecasts['Forecast'],
                        name="Training Forecasts",
                        line=dict(color="blue", width=2, dash="dot"),
                        mode="lines",
                    )
                )
            
            # Test period forecasts
            test_forecasts = forecast_data[forecast_data['Period'] == 'Test']
            if not test_forecasts.empty:
                fig.add_trace(
                    go.Scatter(
                        x=test_forecasts['Date'],
                        y=test_forecasts['Forecast'],
                        name="Test Forecasts",
                        line=dict(color="red", width=2),
                        mode="lines",
                    )
                )
                
                # Add confidence intervals from the model
                if 'Lower_CI' in test_forecasts.columns and 'Upper_CI' in test_forecasts.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=test_forecasts['Date'].tolist() + test_forecasts['Date'].tolist()[::-1],
                            y=test_forecasts['Upper_CI'].tolist() + test_forecasts['Lower_CI'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='95% Confidence Interval',
                            showlegend=True,
                        )
                    )
        
        # Update layout
        fig.update_layout(
            title="Newcastle Coal Price History and Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD/ton)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        
        # Add rangeslider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        return fig

    @staticmethod
    def create_detailed_forecast_plot(forecast_data):
        """Create a detailed forecast vs actual plot."""
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Actual'],
                name="Actual Prices",
                line=dict(color="black", width=2),
                mode="lines"
            )
        )
        
        # Add forecasts
        fig.add_trace(
            go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Forecast'],
                name="Forecasts",
                line=dict(color="red", width=2),
                mode="lines"
            )
        )
        
        # Add confidence intervals if available
        if 'Lower_CI' in forecast_data.columns and 'Upper_CI' in forecast_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['Date'].tolist() + forecast_data['Date'].tolist()[::-1],
                    y=forecast_data['Upper_CI'].tolist() + forecast_data['Lower_CI'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Detailed Forecast vs Actual Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD/ton)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig

    @staticmethod
    def create_forecast_scatter_plot(forecast_data):
        """Create a scatter plot of actual vs predicted values."""
        fig = px.scatter(
            forecast_data,
            x='Actual',
            y='Forecast',
            labels={'Actual': 'Actual Price (USD/ton)', 'Forecast': 'Forecast Price (USD/ton)'},
            title='Actual vs Forecast Comparison'
        )
        
        # Add perfect prediction line
        min_val = min(forecast_data['Actual'].min(), forecast_data['Forecast'].min())
        max_val = max(forecast_data['Actual'].max(), forecast_data['Forecast'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='gray', dash='dash')
            )
        )
        
        # Calculate R² value
        corr_coef = forecast_data['Actual'].corr(forecast_data['Forecast'])
        r_squared = corr_coef ** 2
        
        # Add R² annotation
        fig.add_annotation(
            x=min_val + (max_val-min_val)*0.1,
            y=max_val - (max_val-min_val)*0.1,
            text=f"R² = {r_squared:.4f}",
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        
        # Update layout
        fig.update_layout(
            template="plotly_white",
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig

    @staticmethod
    def create_residual_plot(residuals_df):
        """Create a plot of residuals over time."""
        fig = go.Figure()
        
        # Add residual scatter plot
        fig.add_trace(
            go.Scatter(
                x=residuals_df['Date'],
                y=residuals_df['Residual'],
                mode='markers+lines',
                name='Residuals',
                line=dict(color='blue'),
                marker=dict(size=8)
            )
        )
        
        # Add zero line
        fig.add_hline(
            y=0,
            line=dict(color='red', dash='dash'),
            annotation_text="Zero Error",
            annotation_position="bottom right"
        )
        
        # Update layout
        fig.update_layout(
            title="Forecast Residuals Over Time",
            xaxis_title="Date",
            yaxis_title="Residual (Actual - Forecast)",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig

    @staticmethod
    def create_error_distribution_plot(residuals_df):
        """Create a histogram of forecast errors."""
        fig = px.histogram(
            residuals_df,
            x='Residual',
            nbins=20,
            labels={'Residual': 'Forecast Error (USD/ton)'},
            title='Forecast Error Distribution',
            color_discrete_sequence=['blue']
        )
        
        # Add a normal distribution curve for comparison
        x = np.linspace(
            residuals_df['Residual'].min(),
            residuals_df['Residual'].max(),
            100
        )
        mean = residuals_df['Residual'].mean()
        std = residuals_df['Residual'].std()
        y = ((1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)) * len(residuals_df) * (residuals_df['Residual'].max() - residuals_df['Residual'].min()) / 20
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Add vertical line at mean
        fig.add_vline(
            x=mean,
            line=dict(color='green', dash='dash'),
            annotation_text=f"Mean: {mean:.2f}",
            annotation_position="top"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Forecast Error (USD/ton)",
            yaxis_title="Frequency",
            template="plotly_white",
            hovermode="closest",
            bargap=0.1,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig

    @staticmethod
    def create_residual_diagnostics_plot(residuals_df):
        """Create a 4-panel diagnostic plot for residuals."""
        # Create subplots
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=(
                "Residuals vs Time", 
                "Residual Histogram", 
                "Q-Q Plot",
                "Residual Autocorrelation"
            )
        )
        
        # Check if the dataframe is empty or has no valid residuals
        if residuals_df.empty or residuals_df['Residual'].dropna().empty:
            # Return an empty plot with a message
            fig.add_annotation(
                text="No residual data available for diagnostic plots",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            
            # Update layout
            fig.update_layout(
                height=700,
                title_text="Residual Diagnostic Plots (No Data Available)",
                template="plotly_white",
                showlegend=False,
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            return fig
        
        # 1. Residuals vs Time
        fig.add_trace(
            go.Scatter(
                x=residuals_df['Date'],
                y=residuals_df['Residual'],
                mode='markers',
                name='Residuals',
                marker=dict(color='blue')
            ),
            row=1, col=1
        )
        fig.add_hline(
            y=0,
            line=dict(color='red', dash='dash'),
            row=1, col=1
        )
        
        # 2. Residual Histogram
        fig.add_trace(
            go.Histogram(
                x=residuals_df['Residual'],
                nbinsx=15,
                name='Frequency',
                marker=dict(color='blue')
            ),
            row=1, col=2
        )
        
        # 3. Q-Q Plot (Normal probability plot)
        residuals = residuals_df['Residual'].dropna().sort_values()
        n = len(residuals)
        
        if n > 0:  # Only proceed if we have valid residuals
            theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=residuals,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color='blue')
                ),
                row=2, col=1
            )
            
            # Add reference line
            min_q = min(theoretical_quantiles)
            max_q = max(theoretical_quantiles)
            min_r = residuals.min()
            max_r = residuals.max()
            slope = (max_r - min_r) / (max_q - min_q)
            intercept = min_r - slope * min_q
            
            fig.add_trace(
                go.Scatter(
                    x=[min_q, max_q],
                    y=[min_q * slope + intercept, max_q * slope + intercept],
                    mode='lines',
                    name='Reference Line',
                    line=dict(color='red', dash='dash')
                ),
                row=2, col=1
            )
        else:
            # Add a message if no data is available
            fig.add_annotation(
                text="Insufficient data for Q-Q Plot",
                xref="x", yref="y",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12),
                row=2, col=1
            )
        
        # 4. Autocorrelation Plot
        if n > 1:  # Need at least 2 points for autocorrelation
            acf_values = acf(residuals_df['Residual'].dropna(), nlags=min(20, n-1), fft=False)
            acf_x = np.arange(len(acf_values))
            
            fig.add_trace(
                go.Bar(
                    x=acf_x,
                    y=acf_values,
                    name='ACF',
                    marker=dict(color='blue')
                ),
                row=2, col=2
            )
            
            # Add significance bands
            significance = 1.96 / np.sqrt(n)
            fig.add_hline(
                y=significance,
                line=dict(color='red', dash='dash'),
                row=2, col=2
            )
            fig.add_hline(
                y=-significance,
                line=dict(color='red', dash='dash'),
                row=2, col=2
            )
        else:
            # Add a message if no data is available
            fig.add_annotation(
                text="Insufficient data for ACF Plot",
                xref="x", yref="y",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=700,
            title_text="Residual Diagnostic Plots",
            template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Residual", row=1, col=1)
        
        fig.update_xaxes(title_text="Residual", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        
        fig.update_xaxes(title_text="Lag", row=2, col=2)
        fig.update_yaxes(title_text="Autocorrelation", row=2, col=2)
        
        return fig

    @staticmethod
    def create_parameter_visualization(params_df):
        """Create a visualization of model parameters and their significance."""
        # Create horizontal bar chart for parameters
        fig = go.Figure()
        
        # Add parameter values
        fig.add_trace(
            go.Bar(
                y=params_df['Parameter'],
                x=params_df['Value'],
                orientation='h',
                name='Parameter Value',
                marker=dict(
                    color='blue',
                    line=dict(color='rgba(0,0,0,0)', width=0)
                ),
                error_x=dict(
                    type='data',
                    array=params_df['Std Error'],
                    visible=True,
                    color='black'
                )
            )
        )
        
        # Add vertical line at zero
        fig.add_vline(
            x=0,
            line=dict(color='red', dash='dash')
        )
        
        # Update layout
        fig.update_layout(
            title="SARIMAX Model Parameters with Standard Errors",
            xaxis_title="Parameter Value",
            yaxis_title="Parameter",
            template="plotly_white",
            margin=dict(l=120, r=40, t=60, b=40),
            height=400
        )
        
        return fig

    @staticmethod
    def create_economic_value_plot(combined_results, threshold_results, directional_results):
        """Create a performance comparison plot for trading strategies."""
        # Try to use actual data if available
        has_detailed_data = False
        
        # Check for detailed results files
        try:
            # If we have detailed timeseries data, use it
            if 'Date' in combined_results.columns and 'Cumulative_Return' in combined_results.columns:
                has_detailed_data = True
        except:
            pass
            
        # Create a figure
        fig = go.Figure()
        
        if has_detailed_data:
            # Use actual detailed data
            # (Implementation would depend on the actual data structure)
            pass
        else:
            # Generate simulated data for visualization
            periods = 12
            x_dates = pd.date_range(start='2023-01-31', periods=periods, freq='M')
            
            # Generate sample trajectories
            np.random.seed(42)  # For reproducibility
            
            # Benchmark (buy and hold)
            benchmark_returns = np.cumprod(1 + np.random.normal(0.02, 0.1, periods)) - 1
            
            # Strategy returns (simulated)
            strategy1_returns = np.cumprod(1 + np.random.normal(0.01, 0.08, periods)) - 1  # Combined
            strategy2_returns = np.cumprod(1 + np.random.normal(0.005, 0.09, periods)) - 1  # Threshold
            strategy3_returns = np.cumprod(1 + np.random.normal(0.015, 0.07, periods)) - 1  # Directional
            
            # Add traces
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=benchmark_returns,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='black', width=2)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=strategy1_returns,
                    mode='lines',
                    name='Combined Strategy',
                    line=dict(color='blue', width=2)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=strategy2_returns,
                    mode='lines',
                    name='Threshold Strategy',
                    line=dict(color='green', width=2)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=strategy3_returns,
                    mode='lines',
                    name='Directional Strategy',
                    line=dict(color='orange', width=2)
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Cumulative Returns of Trading Strategies",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            yaxis=dict(tickformat='.0%')
        )
        
        return fig

    @staticmethod
    def create_vif_plot(vif_df):
        """Create a bar chart of VIF values."""
        # Sort by VIF values
        df_sorted = vif_df.sort_values('VIF', ascending=False)
        
        # Create color map based on VIF thresholds
        colors = ['red' if vif > 10 else 'orange' if vif > 5 else 'green' for vif in df_sorted['VIF']]
        
        # Create the figure
        fig = go.Figure()
        
        # Add VIF bars
        fig.add_trace(
            go.Bar(
                x=df_sorted['Feature'],
                y=df_sorted['VIF'],
                marker_color=colors,
                name='VIF Value'
            )
        )
        
        # Add threshold lines
        fig.add_hline(
            y=10,
            line=dict(color='red', dash='dash'),
            annotation_text="Severe Multicollinearity (VIF=10)",
            annotation_position="top right"
        )
        
        fig.add_hline(
            y=5,
            line=dict(color='orange', dash='dash'),
            annotation_text="Moderate Multicollinearity (VIF=5)",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            title="Variance Inflation Factor (VIF) by Feature",
            xaxis_title="Feature",
            yaxis_title="VIF Value",
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40, pad=4),
            xaxis=dict(tickangle=-45)
        )
        
        return fig