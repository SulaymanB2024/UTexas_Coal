# Coal Price Forecasting Project: Final Report

## 1. Introduction

This project aims to develop and evaluate an enhanced time series forecasting model for coal prices with a focus on the Newcastle FOB 6000 NAR benchmark. The primary objective is to improve forecasting accuracy over existing ARIMAX models by implementing and optimizing SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) models. Additionally, the project evaluates the economic value of the forecasts through trading strategy simulations to assess practical utility beyond statistical accuracy.

The scope of this project includes:
- Data collection from reliable energy market sources
- Feature engineering and selection, with a focus on reducing multicollinearity
- Implementation and optimization of SARIMAX models
- Rigorous model diagnostics and evaluation
- Economic value assessment through trading strategy simulation
- Scenario analysis planning for robust decision-making

## 2. Data

### 2.1 Data Sources

The primary data source for this project is the Energy Information Administration (EIA) API, which provides access to comprehensive coal price data and related energy market indicators. The dataset includes:

- Newcastle FOB 6000 NAR coal price (target variable)
- Henry Hub natural gas spot prices
- Baltic Dry Index (shipping cost indicator)
- Various related energy market indicators

### 2.2 Data Preparation

The data preparation process involved several key steps:

1. **Data Integration**: Monthly data was collected from the EIA API and aligned to ensure consistent timeframes.

2. **Missing Value Treatment**: Missing values were identified and imputed using median values to maintain data integrity without introducing extreme values.

3. **Outlier Detection**: Outliers were identified using statistical methods and treated to minimize their impact on model performance.

4. **Data Transformation**: Several transformations were applied to improve feature distributions:
   - Log transformations for price variables
   - Standardization of features to ensure comparability
   - Differencing for achieving stationarity where required

5. **Train-Test Split**: The data was split into training (January 2020 to December 2022) and testing (January 2023 to December 2023) sets, maintaining the temporal structure essential for time series analysis.

## 3. Feature Engineering

### 3.1 Feature Creation Methods

Multiple feature engineering techniques were applied to capture various aspects of the time series data:

1. **Lag Features**: Created lags of the target variable (1, 2, and 3 months) to capture autoregressive patterns and serial dependencies.

2. **Rolling Statistics**: Computed rolling means and standard deviations over different windows (3 and 6 months) to capture trend and volatility patterns.

3. **Momentum Indicators**: Calculated month-over-month changes and moving averages of these changes to capture price momentum.

4. **Transformations**: Applied logarithmic transformations to price variables to improve normality.

5. **Scaling**: Standardized certain variables to improve model convergence.

### 3.2 Feature Selection

A variance inflation factor (VIF) analysis was conducted to address multicollinearity among the engineered features. Features with high VIF values were iteratively removed until all remaining features had VIF values below the threshold of 5.

The final set of 5 low-VIF features selected for the model includes:

1. Henry_Hub_Spot
2. Newcastle_FOB_6000_NAR_roll_std_3
3. Newcastle_FOB_6000_NAR_mom_change
4. Newcastle_FOB_6000_NAR_mom_change_3m_avg
5. Baltic_Dry_Index_scaled

Detailed VIF analysis results are available in the `reports/vif_analysis_results.csv` file.

## 4. Model Selection

### 4.1 Model Development Process

The model development process involved systematic comparison of different time series modeling approaches:

1. **Baseline ARIMAX Models**: Initial models with non-seasonal components were developed as a baseline.

2. **SARIMAX Grid Search**: A comprehensive grid search was performed over different combinations of seasonal and non-seasonal parameters:
   - AR terms (p): 0-2
   - Differencing (d): 0-1
   - MA terms (q): 0-2
   - Seasonal AR terms (P): 0-1
   - Seasonal differencing (D): 0-1
   - Seasonal MA terms (Q): 0-1
   - Seasonal period (m): 12 months

3. **Model Evaluation**: Models were evaluated based on information criteria (AIC, BIC) and diagnostic tests on residuals.

### 4.2 Final Model Specification

The final selected model is a **SARIMAX(1,1,1)(1,0,1,12)** model with the following parameters:
- Non-seasonal AR(1), differencing(1), MA(1)
- Seasonal AR(1), no seasonal differencing, MA(1), period=12 months
- Exogenous variables: The 5 low-VIF features identified in feature selection

The final model's information criteria values are:
- AIC: 151.1095
- BIC: 161.5548

This model specification was chosen as the final accepted model after comprehensive evaluation of various configurations, balancing statistical fit with practical forecasting performance.

## 5. Final Model Diagnostics

### 5.1 Residual Analysis

Comprehensive diagnostic tests were performed on the model residuals to assess model adequacy:

1. **Ljung-Box Test for Autocorrelation**:
   - Lag 10: Test statistic = 5.3259, p-value = 0.8684
   - Lag 20: Test statistic = 9.4676, p-value = 0.9768
   - Lag 30: Test statistic = 11.8077, p-value = 0.9988

   The high p-values (>> 0.05) confirm that residuals are uncorrelated, indicating the model has adequately captured the time series dynamics. This is a significant positive outcome, as it confirms the model's ability to extract the temporal dependencies in the data.

2. **Shapiro-Wilk Test for Normality**:
   - p-value ≈ 0.0000

   The low p-value confirms that the residuals are non-normal, suggesting some deviation from the ideal model assumptions. While this is a limitation, non-normality in financial time series residuals is common and does not necessarily invalidate the model's usefulness for forecasting.

3. **Residual Statistics**:
   - Mean: 4.2809
   - Standard Deviation: 18.6959
   - Min: -17.2427
   - Max: 99.6795

The complete model diagnostic results are available in the `reports/sarimax_fixed_features_summary.txt` file, and visual diagnostics are in the `reports/figures/model_diagnostics.html` file.

## 6. Performance Evaluation

### 6.1 Training Set Performance

Performance metrics for the training period (January 2020 to December 2022):

- **R²**: -0.9388 (negative, suggesting poor in-sample fit)
- **RMSE**: 18.9249
- **MAE**: 10.0590
- **MAPE**: 0.0950 (9.5% error)

The negative R² indicates that the model performs worse than a simple mean model on the training data, which is a significant limitation of this model.

### 6.2 Test Set Performance

Performance metrics for the test period (January 2023 to December 2023):

- **R²**: 0.3302 (model explains 33% of variance in test data)
- **RMSE**: 11.0654
- **MAE**: 9.4886
- **MAPE**: 0.0766 (7.66% error)

While the test set performance is better than the training set performance, the R² of 0.33 indicates only modest predictive power. However, the MAPE of 7.66% suggests reasonably accurate percentage-based predictions.

### 6.3 Performance Interpretation

The model demonstrates better out-of-sample performance than in-sample performance, which is unusual but can occur with certain feature combinations. The negative R² in the training set indicates that the model fits the training data worse than a simple mean model, but the positive R² in the test set suggests it has some predictive value for future data points. The MAPE of 7.66% on the test set is reasonably good for coal price forecasting, indicating that on average, the model's predictions are within 7.66% of the actual values.

## 7. Economic Value Analysis

### 7.1 Trading Strategy Overview

Three trading strategies were implemented to assess the economic value of the SARIMAX model forecasts:

1. **Threshold Strategy**: Takes positions when forecast deviations exceed a threshold calibrated as 0.5 × residual standard deviation.
2. **Directional Strategy**: Takes positions based purely on the predicted direction of price movement.
3. **Combined Strategy**: Takes positions only when both threshold and directional signals agree.

Each strategy was compared against a buy-and-hold benchmark over the test period (January 2023 to December 2023).

### 7.2 Strategy Performance Results

#### 7.2.1 Directional Strategy Results

- **Total Return**: -22.22% (vs Buy-and-Hold: +15.66%)
- **Annualized Return**: -23.98% (vs Buy-and-Hold: +17.20%)
- **Sharpe Ratio**: -0.53 (vs Buy-and-Hold: +0.32)
- **Max Drawdown**: -47.36% (vs Buy-and-Hold: -22.47%)
- **Win Rate**: 45.45% (vs Buy-and-Hold: 54.55%)
- **Trade Statistics**:
  - Total Trades: 4
  - Winning Trades: 1 (25.00%)
  - Losing Trades: 3 (75.00%)
  - Average Holding Period: 76.5 days
  - Average P&L per trade: -$10.95

#### 7.2.2 Threshold Strategy Results

- No trading signals were generated as no forecast deviations exceeded the threshold
- All performance metrics are zero due to no positions being taken

#### 7.2.3 Combined Strategy Results

- No trading signals were generated as the threshold component never triggered
- All performance metrics are zero due to no positions being taken

### 7.3 Economic Value Conclusions

The economic value analysis reveals that neither the directional nor the threshold strategy added economic value compared to a simple buy-and-hold approach during the test period. The directional strategy substantially underperformed the benchmark, while the threshold strategy was too conservative and generated no trades. This is a significant finding that highlights the gap between statistical accuracy and economic utility of the forecasts.

These results clearly demonstrate that even with reasonable statistical metrics (test R² ≈ 0.33, MAPE ≈ 7.66%), the model's forecasts failed to translate into profitable trading strategies that could outperform a passive buy-and-hold approach.

The analysis suggests that:

1. The SARIMAX model forecasts may not be capturing sufficient meaningful deviation from actual prices to support profitable trading.
2. The threshold calibration (0.5 × residual standard deviation) may be too high for the observed forecast errors.
3. Coal price movements during the test period may have exhibited a general upward trend that favored the buy-and-hold strategy.

Detailed results are available in the following files:
- `reports/economic_value_directional_results.csv`
- `reports/economic_value_threshold_results.csv`
- `reports/economic_value_combined_results.csv`

## 8. Scenario Analysis

### 8.1 Scenario Analysis Plan

As part of Iteration 4 of the project plan, a Monte Carlo scenario analysis was planned to evaluate the model's performance under various simulated future market conditions. This approach would involve:

1. Generating multiple simulated price paths based on historical patterns and volatility
2. Applying the SARIMAX model to each simulated path
3. Assessing model robustness across different potential futures
4. Identifying conditions under which the model performs well or poorly

### 8.2 Implementation Status

The implementation of the scenario analysis has been deferred pending review of the current model performance. Given the model's mixed performance in standard evaluation metrics and economic value assessment, it is prudent to first refine the core model before proceeding with extensive scenario testing.

Some preliminary scenario analysis results are available in the following files:
- `reports/scenario_analysis_mc_results.csv`
- `reports/scenario_analysis_mc_sample_paths.csv`
- `reports/figures/scenario_analysis_mc_forecast_paths.html`

## 9. Code Structure

The project follows a modular and well-organized code structure:

```
UTexas_Coal/
├── config/
│   └── config.yaml            # Configuration parameters
├── data/
│   ├── interim/               # Intermediate data files
│   ├── processed/             # Final processed datasets
│   └── raw/                   # Original data files
├── models/                    # Saved model files and outputs
├── reports/                   # Reports and visualizations
│   ├── figures/               # Generated graphics
│   └── markdown/              # Markdown documentation
├── scripts/                   # Executable scripts
│   ├── analyze_multicollinearity.py
│   ├── evaluate_all_strategies.py
│   ├── evaluate_economic_value.py
│   ├── evaluate_fixed_features_sarimax.py
│   ├── evaluate_sarimax.py
│   ├── run_pipeline.py
│   └── run_scenario_analysis.py
└── src/                       # Source code modules
    ├── data_processing/       # Data processing utilities
    ├── modeling/              # Model implementations
    └── utils/                 # General utilities
```

The code design follows these principles:
1. **Modularity**: Functions and classes are designed with single responsibilities
2. **Configurability**: Parameters are externalized in configuration files
3. **Reproducibility**: Random seeds are fixed and all processing steps are documented
4. **Testability**: Functions are designed to be easily tested
5. **Extensibility**: New models or features can be added without changing the core structure

## 10. Deployment Plan

### 10.1 Deployment Strategy

The deployment plan for the coal price forecasting model consists of the following phases:

1. **Model Packaging**:
   - Package the SARIMAX model and preprocessing pipeline
   - Create documentation for API usage and parameter settings
   - Implement versioning for model tracking

2. **API Development**:
   - Develop a RESTful API for model predictions
   - Implement authentication and access controls
   - Create endpoints for batch and real-time predictions

3. **Infrastructure Setup**:
   - Set up cloud-based deployment environment
   - Implement monitoring for system health
   - Configure auto-scaling for handling variable loads

4. **Integration**:
   - Develop client libraries for common programming languages
   - Create integration guides for existing systems
   - Set up regular data refresh pipelines

5. **Monitoring and Maintenance**:
   - Implement model performance monitoring
   - Set up automated retraining schedules
   - Create alert systems for model drift detection

### 10.2 Production Considerations

For production deployment, several additional considerations will be addressed:

1. **Performance Optimization**: Further model optimization to reduce prediction latency
2. **Security**: Implementation of security best practices for data and API access
3. **Scalability**: Ensuring the system can handle increased prediction requests
4. **Availability**: Setting up redundancy for high availability
5. **Documentation**: Comprehensive user and developer documentation

## 11. Conclusion

### 11.1 Summary of Findings

This project developed and evaluated a SARIMAX(1,1,1)(1,0,1,12) model with 5 low-VIF features for forecasting Newcastle FOB 6000 NAR coal prices. The model demonstrated:

- Strong performance in handling autocorrelation, with Ljung-Box tests confirming uncorrelated residuals
- Modest predictive performance on test data (R² of 0.33, MAPE of 7.66%)
- Poor in-sample fit (R² of -0.94), indicating limitations in capturing training data patterns
- Non-normal residuals, suggesting deviations from ideal model assumptions
- Limited economic value, with trading strategies based on the forecasts failing to outperform a simple buy-and-hold benchmark

These mixed results suggest that while the model successfully addresses certain aspects of time series modeling (autocorrelation), it falls short in others (normality, in-sample fit, economic value). The model captures some aspects of coal price dynamics but requires further refinement to improve both statistical accuracy and economic utility.

### 11.2 Limitations

Key limitations identified during the project include:

- Poor in-sample performance (negative R²) despite reasonable out-of-sample metrics
- Limited test set explanatory power (R² of only 0.33)
- Non-normal residuals violating key model assumptions
- Inability to generate economically valuable forecasts for trading
- Divergence between statistical accuracy measures and practical utility
- Challenges with model convergence during estimation
- Limited training data availability

### 11.3 Future Work

Several directions for future work have been identified:

1. **Model Enhancements**:
   - Explore alternative models (e.g., neural networks, hybrid approaches)
   - Incorporate additional relevant exogenous variables
   - Investigate time-varying parameter models

2. **Feature Engineering Improvements**:
   - Develop more sophisticated features capturing market dynamics
   - Explore dimensionality reduction techniques
   - Investigate alternative approaches to handling multicollinearity

3. **Trading Strategy Refinement**:
   - Test lower threshold values for signal generation
   - Explore alternative position sizing approaches
   - Investigate asymmetric trading rules

4. **Extended Validation**:
   - Conduct backtesting over longer historical periods
   - Implement forward validation with new data as it becomes available
   - Complete the planned scenario analysis

5. **Production Implementation**:
   - Develop a web-based dashboard for interactive forecasting
   - Implement continuous model updating
   - Create an alert system for significant forecast changes