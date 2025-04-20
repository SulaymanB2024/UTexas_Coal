"""
Coal Price Forecasting Dashboard
--------------------------------
A dashboard for visualizing and analyzing coal price forecasts 
using an ARIMAX model with exogenous variables.
"""
import os
import dash
from dash import dcc, html
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime

# Try to import dash-bootstrap-components, but provide a fallback if not available
try:
    import dash_bootstrap_components as dbc
    has_bootstrap = True
except ImportError:
    has_bootstrap = False
    print("Warning: dash-bootstrap-components not found. Using basic styling.")

# Import custom components
try:
    from dashboard.components.data_loader import DataLoader
    from dashboard.components.visualizations import Visualizations
    from dashboard.components.ui_components import UIComponents
    from dashboard.layouts.tab_layouts import TabLayouts
    custom_components_available = True
except ImportError:
    custom_components_available = False
    print("Warning: Custom components not found. Using simplified dashboard.")

# Initialize the Dash app with Bootstrap if available
if has_bootstrap:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True
    )
else:
    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True
    )

# Set app title
app.title = "Coal Price Forecasting Dashboard"

# Create a simplified dashboard if custom components are not available
if not custom_components_available:
    app.layout = html.Div([
        html.H1("Coal Price Forecasting Dashboard", style={'textAlign': 'center', 'margin': '20px'}),
        html.Div([
            html.H2("Project Status: Finalized (April 2025)", style={'textAlign': 'center'}),
            html.P("This dashboard would normally show interactive visualizations of the coal price forecasting model.", 
                  style={'textAlign': 'center', 'margin': '20px'}),
            html.P("To enable full functionality, please ensure all required dependencies are installed:", 
                  style={'textAlign': 'center'}),
            html.Ul([
                html.Li("dash"),
                html.Li("dash-bootstrap-components"),
                html.Li("plotly"),
                html.Li("pandas"),
                html.Li("numpy")
            ], style={'width': '200px', 'margin': '0 auto'}),
            html.P([
                "The project has been successfully finalized. The ARIMAX model with VIF-based feature selection ",
                "provides coal price forecasts with a test set MAPE of 7.66% and R² of 0.33."
            ], style={'textAlign': 'center', 'margin': '20px'}),
            html.P([
                "For more information, please refer to the final project report in the reports/markdown directory."
            ], style={'textAlign': 'center', 'margin': '20px'})
        ], style={'margin': '40px', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
    ])
else:
    # Initialize the data loader and visualizations
    data_loader = DataLoader()
    visualizations = Visualizations()

    # Define header layout
    header = html.Div([
        html.Div([
            html.Img(src="/assets/logo.svg", className="logo"),
            html.Div([
                html.H2("Coal Price Forecasting Dashboard", className="app-title"),
                html.P("ARIMAX Model Analysis and Economic Evaluation", className="app-subtitle")
            ], className="title-container")
        ], className="header-content")
    ], className="header")

    # Define the app layout with tabs
    app.layout = html.Div([
        # Header section
        header,
        
        # Main content section with tabs
        html.Div([
            dcc.Tabs(id="tabs", value="tab-overview", className="tabs", children=[
                dcc.Tab(
                    label="Overview",
                    value="tab-overview",
                    className="tab",
                    selected_className="tab-selected"
                ),
                dcc.Tab(
                    label="Forecast Analysis",
                    value="tab-forecast-analysis",
                    className="tab",
                    selected_className="tab-selected"
                ),
                dcc.Tab(
                    label="Model Diagnostics",
                    value="tab-model-diagnostics",
                    className="tab",
                    selected_className="tab-selected"
                ),
                dcc.Tab(
                    label="Economic Value",
                    value="tab-economic-value",
                    className="tab",
                    selected_className="tab-selected"
                ),
                dcc.Tab(
                    label="Feature Analysis",
                    value="tab-feature-analysis",
                    className="tab",
                    selected_className="tab-selected"
                )
            ]),
            html.Div(id="tab-content", className="tab-content")
        ], className="content-container"),
        
        # Footer
        html.Footer([
            html.P(
                ["Coal Price Forecasting Model • ",
                html.Span(f"Last updated: {datetime.now().strftime('%B %d, %Y')}")],
                className="footer-text"
            ),
            html.P(
                "University of Texas at Austin, Energy and Earth Resources Program",
                className="footer-text"
            )
        ], className="footer")
    ], className="app-container")

    # Callback to render tab content
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value")
    )
    def render_tab_content(tab):
        """Render content for the selected tab."""
        if tab == "tab-overview":
            return TabLayouts.render_overview_tab(data_loader, visualizations)
        elif tab == "tab-forecast-analysis":
            return TabLayouts.render_forecast_analysis_tab(data_loader, visualizations)
        elif tab == "tab-model-diagnostics":
            return TabLayouts.render_model_diagnostics_tab(data_loader, visualizations)
        elif tab == "tab-economic-value":
            return TabLayouts.render_economic_value_tab(data_loader, visualizations)
        elif tab == "tab-feature-analysis":
            return TabLayouts.render_feature_analysis_tab(data_loader, visualizations)
        else:
            return html.Div([
                html.H3("Tab content not available", className="error-title"),
                html.P("The selected tab content could not be loaded.", className="error-message")
            ])

# Error handling for the entire app
@app.errorhandler
def handle_error(e):
    print(f"Error: {str(e)}")
    return html.Div([
        html.H3("An error occurred", className="error-title"),
        html.P(f"Error details: {str(e)}", className="error-message"),
        html.Button("Reload", id="reload-button", className="reload-button")
    ])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)