import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from src.data_processing.data_processor import DataProcessor

class ModelEvaluator:
    def __init__(self, y_true, y_pred, pred_intervals=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.pred_intervals = pred_intervals
    
    def calculate_basic_metrics(self):
        """Calculate standard regression metrics"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            'r2': r2_score(self.y_true, self.y_pred),
            'mape': mean_absolute_percentage_error(self.y_true, self.y_pred) * 100
        }
        
        # Calculate directional accuracy
        actual_direction = np.sign(np.diff(self.y_true))
        pred_direction = np.sign(np.diff(self.y_pred))
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        metrics['directional_accuracy'] = directional_accuracy
        
        return metrics
    
    def evaluate_prediction_intervals(self):
        """Evaluate the calibration of prediction intervals"""
        if self.pred_intervals is None:
            return None
            
        lower_bound = self.pred_intervals['lower']
        upper_bound = self.pred_intervals['upper']
        
        # Calculate coverage probability
        in_interval = (self.y_true >= lower_bound) & (self.y_true <= upper_bound)
        coverage_prob = np.mean(in_interval) * 100
        
        # Calculate average interval width
        interval_width = np.mean(upper_bound - lower_bound)
        
        return {
            'coverage_probability': coverage_prob,
            'avg_interval_width': interval_width
        }
    
    def plot_forecasts_with_intervals(self, title="Forecasts with Prediction Intervals"):
        """Plot the forecasts, actual values, and prediction intervals"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_true, label='Actual', color='black')
        plt.plot(self.y_pred, label='Forecast', color='blue')
        
        if self.pred_intervals is not None:
            plt.fill_between(
                range(len(self.y_true)),
                self.pred_intervals['lower'],
                self.pred_intervals['upper'],
                color='blue',
                alpha=0.2,
                label='90% Prediction Interval'
            )
            
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Coal Price')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_path = os.path.join('reports', 'figures', 'forecast_evaluation.png')
        plt.savefig(output_path)
        plt.close()

def main():
    try:
        # Load the processed test data
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_processor = DataProcessor()
        test_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/processed_test.csv'))
        test_data['Date'] = pd.to_datetime(test_data['Date'])
        
        # Load the model predictions
        pred_path = os.path.join(PROJECT_ROOT, 'models/ml_predictions.csv')
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Predictions file not found at {pred_path}")
            
        predictions = pd.read_csv(pred_path)
        
        # Validate required columns exist
        if 'Newcastle_FOB_6000_NAR' not in test_data.columns:
            raise ValueError("Target column 'Newcastle_FOB_6000_NAR' not found in test data")
        required_cols = ['arimax_forecast', 'arimax_lower', 'arimax_upper']
        missing_cols = [col for col in required_cols if col not in predictions.columns]
        if missing_cols:
            raise ValueError(f"Missing required prediction columns: {missing_cols}")
        
        # Initialize evaluator with actual and predicted values
        evaluator = ModelEvaluator(
            test_data['Newcastle_FOB_6000_NAR'].values,
            predictions['arimax_forecast'].values,
            {
                'lower': predictions['arimax_lower'].values,
                'upper': predictions['arimax_upper'].values
            }
        )
        
        # Calculate metrics
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
        output_dir = os.path.join(PROJECT_ROOT, 'reports', 'figures')
        os.makedirs(output_dir, exist_ok=True)
        evaluator.plot_forecasts_with_intervals()
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()