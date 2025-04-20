import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import logging
import os
import sys
from typing import Dict, Any, Tuple

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.utils.logging_config import setup_logging
from src.modeling.base_model import BaseModel
from src.modeling.ts_models import TSModels

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

class MLEnsemble(BaseModel):
    """Machine Learning model ensemble for coal price forecasting."""
    
    def __init__(self, config):
        """Initialize the model ensemble."""
        super().__init__()
        self.config = config
        self.models = {}
        self.best_model = None
        
    def fit(self, X_train, y_train):
        """Fit model ensemble."""
        try:
            # Initialize and fit ARIMAX model
            arimax = TSModels(self.config)
            success = arimax.fit(X_train, y_train)
            
            if not success:
                logger.error("Failed to fit ARIMAX model")
                return False
                
            self.models['arimax'] = arimax
            self.best_model = arimax
            
            logger.info("Successfully fitted ARIMAX model")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting model ensemble: {e}")
            return False
            
    def predict(self, X):
        """Generate predictions and uncertainty estimates from best model.
        
        Returns:
            tuple: (predictions, prediction_std) containing point predictions and their uncertainties
        """
        if self.best_model is None:
            logger.error("No models fitted. Call fit() first.")
            return None, None
            
        try:
            predictions, prediction_std = self.best_model.predict(X)
            return predictions, prediction_std
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return None, None
            
    def get_feature_importance(self):
        """Get feature importance from best model if available."""
        if self.best_model is None:
            logger.error("No models fitted. Call fit() first.")
            return None
            
        try:
            if hasattr(self.best_model, 'model') and hasattr(self.best_model.model, 'params'):
                # Get exogenous variable coefficients from ARIMAX
                coef = pd.Series(
                    self.best_model.model.params[len(self.best_model.model.arparams):],
                    index=self.best_model.model.exog_names
                ).abs()
                
                return coef.sort_values(ascending=False)
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None

if __name__ == '__main__':
    from src.utils.logging_config import load_config, CONFIG_PATH
    
    # Load configuration
    config = load_config(CONFIG_PATH)
    if config is None:
        logger.critical("Failed to load configuration. Exiting.")
        sys.exit(1)
        
    # Example usage (assuming data is available)
    model = MLEnsemble(config)