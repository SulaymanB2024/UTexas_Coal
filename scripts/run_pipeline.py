import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Add project root and src directory to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from src.data_processing.data_processor import DataProcessor
from src.modeling.ml_ensemble import MLEnsemble
from src.utils.logging_config import setup_logging, load_config, CONFIG_PATH

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def run_pipeline():
    """Run the ML modeling pipeline."""
    try:
        # Load configuration
        config = load_config(CONFIG_PATH)
        if config is None:
            logger.critical("Failed to load configuration. Exiting.")
            sys.exit(1)
            
        # Initialize components
        data_processor = DataProcessor(config)
        ml_model = MLEnsemble(config)
        
        # Set up paths
        raw_data_path = os.path.join(PROJECT_ROOT, config['data_paths']['raw'])
        processed_data_path = os.path.join(PROJECT_ROOT, config['data_paths']['processed'])
        model_output_path = os.path.join(PROJECT_ROOT, 'models')
        
        # Create output directories if they don't exist
        os.makedirs(processed_data_path, exist_ok=True)
        os.makedirs(model_output_path, exist_ok=True)
        
        # Load and process raw data
        logger.info("Starting data processing...")
        if not os.path.exists(raw_data_path):
            logger.error(f"Raw data directory not found: {raw_data_path}")
            return False
            
        # Load raw data first
        raw_data = None
        for file in os.listdir(raw_data_path):
            if file.endswith('.csv'):
                logger.info(f"Loading {file}")
                file_path = os.path.join(raw_data_path, file)
                temp_data = pd.read_csv(file_path)
                temp_data[config['variables']['date_column']] = pd.to_datetime(temp_data[config['variables']['date_column']])
                temp_data.set_index(config['variables']['date_column'], inplace=True)
                
                if raw_data is None:
                    raw_data = temp_data
                else:
                    raw_data = raw_data.join(temp_data, how='outer')
        
        if raw_data is None:
            logger.error("No data was loaded")
            return False
            
        # Split into training and testing sets first
        train_end = pd.Timestamp(config['modeling']['train_end_date'])
        train_data = raw_data[raw_data.index <= train_end].copy()
        test_data = raw_data[raw_data.index > train_end].copy()
        
        # Process training data
        logger.info("Processing training data...")
        data_processor.data = train_data
        data_processor.clean_data()
        data_processor.create_features()
        data_processor.transform_data()
        processed_train = data_processor.data.copy()
        
        # Process test data using same transformations
        logger.info("Processing test data...")
        data_processor.data = test_data
        data_processor.clean_data()
        data_processor.create_features()
        data_processor.transform_data()
        processed_test = data_processor.data.copy()
        
        # Save processed datasets
        train_file = os.path.join(processed_data_path, "processed_train.csv")
        test_file = os.path.join(processed_data_path, "processed_test.csv")
        processed_train.to_csv(train_file)
        processed_test.to_csv(test_file)
        
        # Prepare features for modeling
        X_train, y_train = data_processor.prepare_data(processed_train)
        X_test, y_test = data_processor.prepare_data(processed_test)
        
        if X_train is None or y_train is None:
            logger.error("Failed to prepare training data")
            return False
            
        # Fit and evaluate ML ensemble
        logger.info("Fitting ML ensemble...")
        if ml_model.fit(X_train, y_train):
            predictions = ml_model.predict(X_test)
            if predictions is not None and len(predictions) == 2:
                ml_predictions, ml_std = predictions
            else:
                logger.error("Invalid prediction output format")
                return False
            
            # Run diagnostic analysis
            logger.info("Running model diagnostics...")
            diagnostic_results = data_processor.analyze_predictions(
                y_test, 
                ml_predictions,
                config['variables']['target_variable']
            )
            
            # Save ML predictions and feature importance
            ml_output_file = os.path.join(model_output_path, 'ml_predictions.csv')
            ml_model.save_predictions(y_test, ml_predictions, ml_output_file)
            
            importance_file = os.path.join(model_output_path, 'feature_importance.csv')
            ml_model.save_feature_importance(importance_file)
            
        logger.info("Pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")
        return False

if __name__ == '__main__':
    logger.info("Starting model pipeline...")
    success = run_pipeline()
    if success:
        logger.info("Pipeline completed successfully")
    else:
        logger.error("Pipeline failed")
        sys.exit(1)