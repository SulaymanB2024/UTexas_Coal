import os
import pandas as pd
import numpy as np
from src.modeling.ts_models import TSModels
from src.data_processing.data_processor import DataProcessor
from scripts.evaluate_model import ModelEvaluator
from src.utils.logging_config import load_config, CONFIG_PATH

def clean_data(df):
    """Clean data by handling NaN and infinite values"""
    # Replace inf/-inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill NaN values first
    df = df.fillna(method='ffill')
    
    # Then backfill any remaining NaNs at the start
    df = df.fillna(method='bfill')
    
    return df

def main():
    try:
        # Load configuration
        config = load_config(CONFIG_PATH)
        if config is None:
            raise ValueError("Failed to load configuration")

        # Initialize models and data processor
        model = TSModels(config)
        data_processor = DataProcessor(config)

        # Load data
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        train_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/processed_train.csv'))
        test_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/processed_test.csv'))
        
        # Convert dates
        train_data['Date'] = pd.to_datetime(train_data['Date'])
        test_data['Date'] = pd.to_datetime(test_data['Date'])
        
        # Set index
        train_data.set_index('Date', inplace=True)
        test_data.set_index('Date', inplace=True)

        # Clean data
        train_data = clean_data(train_data)
        test_data = clean_data(test_data)

        # Prepare features and target
        target_col = 'Newcastle_FOB_6000_NAR'
        feature_cols = [col for col in train_data.columns if col != target_col]

        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]

        # Fit model
        print("Fitting ARIMAX model...")
        success = model.fit(X_train, y_train)
        if not success:
            raise RuntimeError("Failed to fit ARIMAX model")

        # Generate predictions with uncertainty
        print("Generating predictions...")
        predictions, pred_std = model.predict(X_test)
        if predictions is None or pred_std is None:
            raise RuntimeError("Failed to generate predictions")

        # Create prediction intervals
        pred_intervals = {
            'lower': predictions - 1.96 * pred_std,  # 95% interval
            'upper': predictions + 1.96 * pred_std
        }

        # Save predictions
        results_df = pd.DataFrame({
            'arimax_forecast': predictions,
            'arimax_lower': pred_intervals['lower'],
            'arimax_upper': pred_intervals['upper']
        })
        results_df.to_csv(os.path.join(PROJECT_ROOT, 'models/ml_predictions.csv'))

        # Evaluate model
        evaluator = ModelEvaluator(y_test.values, predictions, pred_intervals)
        metrics = evaluator.calculate_basic_metrics()
        interval_metrics = evaluator.evaluate_prediction_intervals()

        # Print results
        print("\nModel Performance Metrics:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R²: {metrics['r2']:.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")

        if interval_metrics:
            print("\nPrediction Interval Evaluation:")
            print(f"Coverage Probability: {interval_metrics['coverage_probability']:.2f}%")
            print(f"Average Interval Width: {interval_metrics['avg_interval_width']:.2f}")

        # Generate plots
        evaluator.plot_forecasts_with_intervals()
        print("\nEvaluation complete. Results saved to models/ml_predictions.csv")
        print("Plots saved to reports/figures/forecast_evaluation.png")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
