import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Tuple

def calculate_vif(data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Calculate VIF for each feature in the dataset.
    
    Args:
        data: DataFrame containing the features
        features: List of feature names to calculate VIF for
        
    Returns:
        DataFrame with VIF scores for each feature
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(data[features].values, i)
                       for i in range(len(features))]
    return vif_data.sort_values('VIF', ascending=False)

def iterative_vif_elimination(data: pd.DataFrame, 
                            initial_features: List[str],
                            threshold: float = 10.0) -> Tuple[List[str], pd.DataFrame]:
    """
    Iteratively remove features with highest VIF until all features are below threshold.
    
    Args:
        data: DataFrame containing the features
        initial_features: List of feature names to start with
        threshold: VIF threshold for feature elimination
        
    Returns:
        Tuple of (final feature list, DataFrame with VIF history)
    """
    current_features = initial_features.copy()
    vif_history = []
    
    while True:
        vif_data = calculate_vif(data, current_features)
        max_vif = vif_data['VIF'].max()
        vif_history.append(vif_data.copy())
        
        if max_vif < threshold:
            break
            
        # Remove feature with highest VIF
        feature_to_remove = vif_data.iloc[0]['Feature']
        current_features.remove(feature_to_remove)
        
        if len(current_features) < 2:
            break
    
    return current_features, pd.concat(vif_history, keys=range(len(vif_history)))