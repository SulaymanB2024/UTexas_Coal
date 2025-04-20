"""
Simplified Coal Price Forecasting Dashboard
------------------------------------------
A basic dashboard showing key project results.
"""
import os
import sys
import dash
from dash import dcc, html
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Coal Price Forecasting Dashboard"

# Define the app layout
app.layout = html.Div([
    # Header
    html.H1("Coal Price Forecasting Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': '20px'}),
    
    # Project Status
    html.Div([
        html.H2("Project Status: Finalized (April 2025)", 
                style={'textAlign': 'center', 'color': '#27ae60'}),
        
        # Project Overview
        html.Div([
            html.H3("Project Overview", style={'color': '#2980b9'}),
            html.P([
                "This project implements a quantitative framework to forecast Newcastle FOB 6000kc NAR coal prices ",
                "using ARIMAX modeling with VIF-based feature selection. The model achieves a test set MAPE of 7.66% ",
                "and R² of 0.33."
            ])
        ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Model Performance
        html.Div([
            html.H3("Model Performance", style={'color': '#2980b9'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Metric", style={'padding': '10px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                    html.Th("Training Set", style={'padding': '10px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                    html.Th("Test Set", style={'padding': '10px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'})
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("R²", style={'padding': '10px', 'border': '1px solid #ddd'}),
                        html.Td("-0.9388", style={'padding': '10px', 'border': '1px solid #ddd', 'color': 'red'}),
                        html.Td("0.3302", style={'padding': '10px', 'border': '1px solid #ddd', 'color': 'green'})
                    ]),
                    html.Tr([
                        html.Td("RMSE", style={'padding': '10px', 'border': '1px solid #ddd'}),
                        html.Td("18.9249", style={'padding': '10px', 'border': '1px solid #ddd'}),
                        html.Td("11.0654", style={'padding': '10px', 'border': '1px solid #ddd'})
                    ]),
                    html.Tr([
                        html.Td("MAE", style={'padding': '10px', 'border': '1px solid #ddd'}),
                        html.Td("10.0590", style={'padding': '10px', 'border': '1px solid #ddd'}),
                        html.Td("9.4886", style={'padding': '10px', 'border': '1px solid #ddd'})
                    ]),
                    html.Tr([
                        html.Td("MAPE", style={'padding': '10px', 'border': '1px solid #ddd'}),
                        html.Td("9.50%", style={'padding': '10px', 'border': '1px solid #ddd'}),
                        html.Td("7.66%", style={'padding': '10px', 'border': '1px solid #ddd'})
                    ])
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Economic Value
        html.Div([
            html.H3("Economic Value Assessment", style={'color': '#2980b9'}),
            html.P([
                "Trading strategies based on the model's predictions were evaluated against a buy-and-hold benchmark."
            ]),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Strategy", style={'padding': '10px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                    html.Th("Total Return", style={'padding': '10px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                    html.Th("Sharpe Ratio", style={'padding': '10px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                    html.Th("Win Rate", style={'padding': '10px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'})
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("Directional", style={'padding': '10px', 'border': '1px solid #ddd'}),
                        html.Td("-22.22%", style={'padding': '10px', 'border': '1px solid #ddd', 'color': 'red'}),
                        html.Td("-0.53", style={'padding': '10px', 'border': '1px solid #ddd', 'color': 'red'}),
                        html.Td("45.45%", style={'padding': '10px', 'border': '1px solid #ddd'})
                    ]),
                    html.Tr([
                        html.Td("Buy-and-Hold", style={'padding': '10px', 'border': '1px solid #ddd'}),
                        html.Td("+15.66%", style={'padding': '10px', 'border': '1px solid #ddd', 'color': 'green'}),
                        html.Td("+0.32", style={'padding': '10px', 'border': '1px solid #ddd', 'color': 'green'}),
                        html.Td("54.55%", style={'padding': '10px', 'border': '1px solid #ddd'})
                    ])
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Key Features
        html.Div([
            html.H3("Selected Features", style={'color': '#2980b9'}),
            html.P([
                "After VIF analysis to reduce multicollinearity, the following features were selected for the final model:"
            ]),
            html.Ul([
                html.Li("Henry_Hub_Spot"),
                html.Li("Newcastle_FOB_6000_NAR_roll_std_3"),
                html.Li("Newcastle_FOB_6000_NAR_mom_change"),
                html.Li("Newcastle_FOB_6000_NAR_mom_change_3m_avg"),
                html.Li("Baltic_Dry_Index_scaled")
            ])
        ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Final Model
        html.Div([
            html.H3("Final Model Specification", style={'color': '#2980b9'}),
            html.P([
                "The final selected model is a SARIMAX(1,1,1)(1,0,1,12) model with the following parameters:",
                html.Br(),
                "- Non-seasonal AR(1), differencing(1), MA(1)",
                html.Br(),
                "- Seasonal AR(1), no seasonal differencing, MA(1), period=12 months",
                html.Br(),
                "- Information criteria: AIC = 151.1095, BIC = 161.5548"
            ])
        ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
        
    ], style={'margin': '40px', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': 'white'}),
    
    # Footer
    html.Footer([
        html.P(
            ["Coal Price Forecasting Model • Last updated: April 20, 2025"],
            style={'textAlign': 'center', 'color': '#7f8c8d'}
        ),
        html.P(
            "University of Texas at Austin, Energy and Earth Resources Program",
            style={'textAlign': 'center', 'color': '#7f8c8d'}
        )
    ], style={'margin': '40px 0', 'padding': '20px'})
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': '0 auto', 'backgroundColor': '#ecf0f1', 'padding': '20px'})

# Run the app
if __name__ == "__main__":
    print("Starting simplified Coal Price Forecasting Dashboard...")
    print("Go to http://127.0.0.1:8050/ in your web browser to view the dashboard")
    app.run_server(debug=True, port=8050)