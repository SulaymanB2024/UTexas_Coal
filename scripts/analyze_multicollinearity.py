import sys
import os
import pandas as pd
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.modeling.multicollinearity_diagnostics import analyze_multicollinearity
from src.utils.logging_config import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load the processed training data
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_train.csv')
    df = pd.read_csv(data_path)
    
    # Drop the target variable and any date/time columns
    X = df.drop(columns=['Price', 'Date'])  # Adjust column names if different
    
    # Run multicollinearity analysis
    selected_features, initial_vif, removed_features = analyze_multicollinearity(
        X, 
        vif_threshold=10.0
    )
    
    # Save results
    reports_dir = os.path.join(project_root, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save VIF analysis results
    results_df = pd.DataFrame({
        'Initial_VIF': initial_vif
    })
    results_df['Removed'] = results_df.index.isin(removed_features.keys())
    results_df['Final_Features'] = results_df.index.isin(selected_features.columns)
    
    results_path = os.path.join(reports_dir, 'vif_analysis_results.csv')
    results_df.to_csv(results_path)
    logger.info(f"VIF analysis results saved to {results_path}")

if __name__ == "__main__":
    main()