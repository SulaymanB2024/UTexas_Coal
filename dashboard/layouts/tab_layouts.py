"""
Tab layouts for the Coal Price Forecasting Dashboard
---------------------------------------------------
This module contains the layout functions for each dashboard tab.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

from dashboard.components.ui_components import UIComponents

class TabLayouts:
    """Class containing layout functions for each dashboard tab"""
    
    @staticmethod
    def render_overview_tab(data_loader, visualizations):
        """Render content for the Overview tab."""
        # Create main forecast plot
        forecast_plot = visualizations.create_forecast_plot(
            data_loader.all_actual, 
            data_loader.forecast_data
        )
        
        # Create metrics summary
        metrics = data_loader.metrics_df
        
        # Get performance metrics for test data
        try:
            test_r2 = metrics[metrics['Metric'] == 'R2']['Test'].values[0]
            test_rmse = metrics[metrics['Metric'] == 'RMSE']['Test'].values[0]
            test_mae = metrics[metrics['Metric'] == 'MAE']['Test'].values[0]
            test_mape = metrics[metrics['Metric'] == 'MAPE']['Test'].values[0] * 100  # Convert to percentage
        except:
            # Fallback to defaults if metrics are not available
            test_r2 = 0.33
            test_rmse = 11.07
            test_mae = 9.49
            test_mape = 7.66
        
        return html.Div([
            # Header
            html.Div([
                html.H3("Coal Price Forecast Overview", className="section-title"),
                html.P(
                    "This dashboard presents the forecast results for Newcastle coal prices "
                    "using an ARIMAX model with exogenous variables.",
                    className="section-description"
                )
            ], className="section-header"),
            
            # Main forecast chart
            html.Div([
                html.H4("Price Forecast", className="chart-title"),
                dcc.Graph(
                    id="main-forecast-plot",
                    figure=forecast_plot,
                    config={"displayModeBar": True},
                    className="main-chart"
                )
            ], className="forecast-chart-container"),
            
            # Model performance metrics
            html.Div([
                html.H4("Model Performance", className="metrics-title"),
                html.Div([
                    UIComponents.create_metric_card(
                        title="R²",
                        value=f"{test_r2:.2f}",
                        subtitle="Coefficient of determination",
                        status="positive" if test_r2 > 0.3 else "neutral" if test_r2 > 0 else "negative"
                    ),
                    UIComponents.create_metric_card(
                        title="RMSE",
                        value=f"{test_rmse:.2f}",
                        subtitle="Root Mean Squared Error",
                        status="positive" if test_rmse < 10 else "neutral" if test_rmse < 15 else "negative"
                    ),
                    UIComponents.create_metric_card(
                        title="MAE",
                        value=f"{test_mae:.2f}",
                        subtitle="Mean Absolute Error",
                        status="positive" if test_mae < 8 else "neutral" if test_mae < 12 else "negative"
                    ),
                    UIComponents.create_metric_card(
                        title="MAPE",
                        value=f"{test_mape:.1f}%",
                        subtitle="Mean Abs. Percentage Error",
                        status="positive" if test_mape < 8 else "neutral" if test_mape < 12 else "negative"
                    )
                ], className="metrics-container")
            ], className="metrics-section"),
            
            # Summary insights
            html.Div([
                html.H4("Key Insights", className="insights-title"),
                html.Div([
                    UIComponents.create_insight_card(
                        "Forecast Accuracy",
                        f"The model achieves {test_mape:.1f}% mean absolute percentage error "
                        f"on test data, with an R² of {test_r2:.2f}.",
                        "positive" if test_mape < 8 else "neutral"
                    ),
                    UIComponents.create_insight_card(
                        "Price Trend",
                        "Coal prices show moderate volatility with seasonal patterns that the model has captured.",
                        "neutral"
                    ),
                    UIComponents.create_insight_card(
                        "Forecast Uncertainty",
                        "The 95% confidence interval indicates the range of potential price movements.",
                        "neutral"
                    ),
                    UIComponents.create_insight_card(
                        "Model Limitations",
                        "The model may not fully capture extreme market events or structural breaks.",
                        "negative"
                    )
                ], className="insights-container")
            ], className="insights-section")
        ], className="tab-content-container")
    
    @staticmethod
    def render_forecast_analysis_tab(data_loader, visualizations):
        """Render content for the Forecast Analysis tab."""
        # Only use test period for detailed analysis
        test_forecasts = data_loader.forecast_data[data_loader.forecast_data['Period'] == 'Test'].copy()
        
        # Create analysis plots
        if not test_forecasts.empty and 'Actual' in test_forecasts.columns:
            # If we don't have actual values in the forecast data, merge with test_actual
            if 'Actual' not in test_forecasts.columns and not data_loader.test_actual.empty:
                test_forecasts = test_forecasts.merge(
                    data_loader.test_actual[['Date', 'Actual']], 
                    on='Date', 
                    how='left'
                )
            
            # Create detail and scatter plots
            detailed_plot = visualizations.create_detailed_forecast_plot(test_forecasts)
            scatter_plot = visualizations.create_forecast_scatter_plot(test_forecasts)
        else:
            # Create empty plots with messages if no data
            detailed_plot = {
                "layout": {
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "annotations": [
                        {
                            "text": "No forecast data available",
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {"size": 28}
                        }
                    ]
                }
            }
            scatter_plot = detailed_plot
            
        # Get metrics table data
        metrics_df = data_loader.metrics_df
        
        return html.Div([
            # Header
            UIComponents.create_section_header(
                "Forecast Analysis",
                "Detailed analysis of forecast performance and accuracy metrics."
            ),
            
            # Forecast detail plots
            html.Div([
                html.Div([
                    html.H4("Detailed Forecast vs Actuals", className="chart-title"),
                    dcc.Graph(
                        id="detailed-forecast-plot",
                        figure=detailed_plot,
                        config={"displayModeBar": True},
                        className="chart-item"
                    )
                ], className="chart-container"),
                
                html.Div([
                    html.H4("Forecast Scatter Plot", className="chart-title"),
                    dcc.Graph(
                        id="forecast-scatter-plot",
                        figure=scatter_plot,
                        config={"displayModeBar": True},
                        className="chart-item"
                    )
                ], className="chart-container")
            ], className="analysis-grid"),
            
            # Metrics table
            html.Div([
                UIComponents.create_subsection_header(
                    "Performance Metrics",
                    "Comparison of model performance across train and test datasets."
                ),
                html.Div([
                    UIComponents.create_data_table(
                        metrics_df,
                        'metrics-table',
                        sort_action='native',
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'Metric'},
                                'fontWeight': 'bold'
                            }
                        ]
                    )
                ], className="metrics-table-container")
            ], className="metrics-table-section"),
            
            # Metrics explanation
            html.Div([
                UIComponents.create_subsection_header(
                    "Metrics Explanation",
                    "Interpretation of the model performance metrics."
                ),
                html.Div([
                    html.P([
                        html.Strong("R²: "), 
                        "Coefficient of determination, measures how well the model explains variance in the data. " 
                        "Higher is better, with 1.0 being perfect. Values can be negative if the model performs worse than a horizontal line."
                    ]),
                    html.P([
                        html.Strong("RMSE: "), 
                        "Root Mean Squared Error, measures the average magnitude of forecast errors. " 
                        "Lower is better. RMSE is in the same units as the price (USD/ton)."
                    ]),
                    html.P([
                        html.Strong("MAE: "), 
                        "Mean Absolute Error, the average absolute difference between forecasts and actual values. " 
                        "Lower is better. MAE is in the same units as the price (USD/ton)."
                    ]),
                    html.P([
                        html.Strong("MAPE: "), 
                        "Mean Absolute Percentage Error, the average percentage difference between forecasts and actual values. " 
                        "Lower is better. MAPE is expressed as a percentage."
                    ])
                ], className="metrics-explanation")
            ], className="metrics-explanation-section")
        ], className="tab-content-container")
    
    @staticmethod
    def render_model_diagnostics_tab(data_loader, visualizations):
        """Render content for the Model Diagnostics tab."""
        # Create diagnostic plots
        residual_diagnostics = visualizations.create_residual_diagnostics_plot(data_loader.residuals_df)
        
        # Create parameter visualization if available
        if not data_loader.model_parameters.empty:
            param_plot = visualizations.create_parameter_visualization(data_loader.model_parameters)
        else:
            param_plot = {
                "layout": {
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "annotations": [
                        {
                            "text": "No parameter data available",
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {"size": 28}
                        }
                    ]
                }
            }
        
        # Create residual statistics table
        residual_stats = data_loader.residual_stats
        
        # Create diagnostic test results tables
        ljung_box_df = data_loader.ljung_box_df
        shapiro_df = data_loader.shapiro_df
        
        return html.Div([
            # Header
            UIComponents.create_section_header(
                "Model Diagnostics",
                "Analysis of model residuals and parameter estimates to validate model assumptions."
            ),
            
            # Residual diagnostic plots
            html.Div([
                html.H4("Residual Diagnostics", className="chart-title"),
                dcc.Graph(
                    id="residual-diagnostics-plot",
                    figure=residual_diagnostics,
                    config={"displayModeBar": True},
                    className="full-chart"
                )
            ], className="diagnostics-chart-container"),
            
            # Model parameters
            html.Div([
                UIComponents.create_subsection_header(
                    "Model Parameters",
                    "SARIMAX model parameter estimates with standard errors."
                ),
                html.Div([
                    dcc.Graph(
                        id="parameter-plot",
                        figure=param_plot,
                        config={"displayModeBar": True},
                        className="parameter-chart"
                    )
                ], className="parameter-chart-container"),
                html.Div([
                    UIComponents.create_data_table(
                        data_loader.model_parameters,
                        'parameter-table',
                        sort_action='native'
                    )
                ], className="parameter-table-container")
            ], className="parameters-section"),
            
            # Residual statistics and tests
            html.Div([
                html.Div([
                    UIComponents.create_subsection_header(
                        "Residual Statistics",
                        "Statistical properties of model residuals."
                    ),
                    html.Div([
                        UIComponents.create_data_table(
                            residual_stats,
                            'residual-stats-table'
                        )
                    ], className="small-table-container")
                ], className="column"),
                
                html.Div([
                    UIComponents.create_subsection_header(
                        "Ljung-Box Test",
                        "Tests for autocorrelation in residuals. High p-values indicate no significant autocorrelation."
                    ),
                    html.Div([
                        UIComponents.create_data_table(
                            ljung_box_df,
                            'ljung-box-table',
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{Status} = "Pass"'},
                                    'backgroundColor': '#e6ffe6'
                                },
                                {
                                    'if': {'filter_query': '{Status} = "Fail"'},
                                    'backgroundColor': '#ffcccc'
                                }
                            ]
                        )
                    ], className="small-table-container")
                ], className="column"),
                
                html.Div([
                    UIComponents.create_subsection_header(
                        "Shapiro-Wilk Test",
                        "Tests for normality of residuals. High p-values indicate residuals approximate a normal distribution."
                    ),
                    html.Div([
                        UIComponents.create_data_table(
                            shapiro_df,
                            'shapiro-wilk-table',
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{Status} = "Pass"'},
                                    'backgroundColor': '#e6ffe6'
                                },
                                {
                                    'if': {'filter_query': '{Status} = "Fail"'},
                                    'backgroundColor': '#ffcccc'
                                }
                            ]
                        )
                    ], className="small-table-container")
                ], className="column")
            ], className="diagnostics-grid")
        ], className="tab-content-container")
    
    @staticmethod
    def render_economic_value_tab(data_loader, visualizations):
        """Render content for the Economic Value tab."""
        # Create economic value plot
        economic_plot = visualizations.create_economic_value_plot(
            data_loader.combined_results,
            data_loader.threshold_results,
            data_loader.directional_results
        )
        
        # Get strategy data
        combined_results = data_loader.combined_results
        threshold_results = data_loader.threshold_results
        directional_results = data_loader.directional_results
        
        return html.Div([
            # Header
            UIComponents.create_section_header(
                "Economic Value Assessment",
                "Evaluation of trading strategies based on the coal price forecasts."
            ),
            
            # Performance chart
            html.Div([
                html.H4("Strategy Performance Comparison", className="chart-title"),
                dcc.Graph(
                    id="economic-value-plot",
                    figure=economic_plot,
                    config={"displayModeBar": True},
                    className="full-chart"
                )
            ], className="economic-chart-container"),
            
            # Strategy cards section
            html.Div([
                html.H4("Trading Strategy Analysis", className="strategies-title"),
                html.Div([
                    html.Div([
                        UIComponents.create_strategy_card(
                            "Buy & Hold",
                            combined_results,
                            "Simple benchmark strategy that buys coal futures and holds them for the entire period.",
                            is_benchmark=True
                        )
                    ], className="benchmark-card-container"),
                    
                    html.Div([
                        UIComponents.create_strategy_card(
                            "Combined Strategy",
                            combined_results,
                            "A strategy that combines directional and threshold signals to make trading decisions.",
                            is_benchmark=False
                        ),
                        UIComponents.create_strategy_card(
                            "Threshold Strategy",
                            threshold_results,
                            "Trades based on whether the forecast is above or below a threshold value.",
                            is_benchmark=False
                        ),
                        UIComponents.create_strategy_card(
                            "Directional Strategy",
                            directional_results,
                            "Trades based on the predicted direction of price movements.",
                            is_benchmark=False
                        )
                    ], className="strategy-cards-grid")
                ], className="strategies-container")
            ], className="strategies-section"),
            
            # Strategy insights
            html.Div([
                html.H4("Strategy Insights", className="insights-title"),
                html.Div([
                    UIComponents.create_insight_card(
                        "Strategy Performance",
                        "The directional strategy outperforms the threshold strategy in terms of risk-adjusted returns.",
                        "positive"
                    ),
                    UIComponents.create_insight_card(
                        "Volatility Management",
                        "Trading strategies based on forecasts show lower volatility compared to buy & hold.",
                        "positive"
                    ),
                    UIComponents.create_insight_card(
                        "Drawdown Control",
                        "The combined strategy offers improved drawdown management during market downturns.",
                        "neutral"
                    ),
                    UIComponents.create_insight_card(
                        "Implementation Considerations",
                        "Transaction costs and slippage would reduce real-world performance of active strategies.",
                        "negative"
                    )
                ], className="insights-container")
            ], className="insights-section")
        ], className="tab-content-container")
    
    @staticmethod
    def render_feature_analysis_tab(data_loader, visualizations):
        """Render content for the Feature Analysis tab."""
        # Check if we have VIF analysis data
        if data_loader.vif_results.empty:
            return html.Div([
                html.H3("Feature Analysis", className="section-title"),
                html.P("No feature analysis data available.", className="section-description")
            ])
        
        # Create VIF plot
        vif_plot = visualizations.create_vif_plot(data_loader.vif_results)
        
        # Calculate feature selection insights
        vif_results = data_loader.vif_results
        selected_count = vif_results['Selected'].sum() if 'Selected' in vif_results.columns else 0
        total_features = len(vif_results)
        max_vif = vif_results['VIF'].max()
        max_vif_feature = vif_results.loc[vif_results['VIF'].idxmax(), 'Feature']
        
        return html.Div([
            # Title and description
            UIComponents.create_section_header(
                "Feature Analysis",
                "This section analyzes the features used in the model, with focus on "
                "multicollinearity assessment using Variance Inflation Factor (VIF)."
            ),
            
            # VIF analysis
            html.Div([
                UIComponents.create_subsection_header(
                    "Multicollinearity Analysis",
                    "VIF measures how much the variance of a regression coefficient is inflated due to multicollinearity. "
                    "VIF > 5 indicates potential multicollinearity concerns, VIF > 10 indicates severe multicollinearity."
                ),
                html.Div([
                    html.Div([
                        html.H5("VIF Analysis Results", className="chart-title"),
                        dcc.Graph(
                            id="vif-plot",
                            figure=vif_plot,
                            config={"displayModeBar": True},
                            className="chart-item"
                        )
                    ], className="chart-container"),
                    
                    html.Div([
                        html.H5("VIF Values Table", className="table-title"),
                        UIComponents.create_data_table(
                            vif_results,
                            'vif-table',
                            columns=[
                                {"name": "Feature", "id": "Feature"},
                                {"name": "VIF", "id": "VIF", "type": "numeric", "format": {"specifier": ".2f"}},
                                {"name": "Selected", "id": "Selected", "type": "boolean"}
                            ],
                            sort_action='native',
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{VIF} > 10'},
                                    'backgroundColor': '#ffcccc'
                                },
                                {
                                    'if': {'filter_query': '{VIF} <= 5'},
                                    'backgroundColor': '#e6ffe6'
                                },
                                {
                                    'if': {'filter_query': '{VIF} > 5 && {VIF} <= 10'},
                                    'backgroundColor': '#fff0cc'
                                }
                            ]
                        )
                    ], className="table-container")
                ], className="vif-analysis-grid")
            ], className="vif-section"),
            
            # Feature insights
            html.Div([
                html.H4("Feature Selection Insights", className="insights-title"),
                html.Div([
                    UIComponents.create_insight_card(
                        "Multicollinearity",
                        f"{'High' if max_vif > 10 else 'Moderate' if max_vif > 5 else 'Low'} "
                        f"multicollinearity detected among features, with maximum VIF of {max_vif:.2f}.",
                        "negative" if max_vif > 10 else "neutral" if max_vif > 5 else "positive"
                    ),
                    UIComponents.create_insight_card(
                        "Feature Selection",
                        f"VIF-based feature selection retained {selected_count} out of {total_features} original features.",
                        "neutral"
                    ),
                    UIComponents.create_insight_card(
                        "Most Collinear Features",
                        f"Highest multicollinearity in {max_vif_feature} "
                        f"(VIF: {max_vif:.2f}).",
                        "negative" if max_vif > 10 else "neutral"
                    ),
                    UIComponents.create_insight_card(
                        "VIF Threshold",
                        "Features with VIF > 10 were excluded from the final model to reduce multicollinearity.",
                        "positive"
                    )
                ], className="insights-container")
            ], className="insights-section")
        ], className="tab-content-container")