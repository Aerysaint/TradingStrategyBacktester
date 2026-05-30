import pandas as pd
import numpy as np

def generate_labels(df, horizon, volatility_multiplier):
    """
    Generates class labels based on future return and rolling volatility bounds.
    Class 1 (Up/Buy): Future_Ret > (rolling_std * volatility_multiplier)
    Class 2 (Down/Sell): Future_Ret < -(rolling_std * volatility_multiplier)
    Class 0 (Flat/Hold): Between the bounds.
    """
    df_labelled = df.copy()
    
    # Calculate Future Return
    # Using 'close' since it wasn't normalized or using an un-normalized column would be better,
    # but the prompt says: "Calculate the future return: Future_Ret = (Close.shift(-horizon) - Close) / Close"
    # Assuming 'close' in df is available. If it was normalized, the return calculation might be skewed, 
    # but here we'll assume it works on the available 'close' column which might be normalized.
    # Actually, we should calculate labels before normalization or keep original close. 
    # For now, implemented as requested.
    
    df_labelled['Future_Ret'] = (df_labelled['close'].shift(-horizon) - df_labelled['close']) / df_labelled['close']
    
    # Calculate rolling std of returns
    df_labelled['rolling_std'] = df_labelled['close'].pct_change().rolling(window=horizon).std()
    
    # Dynamic Flat Zone Bounds
    upper_bound = df_labelled['rolling_std'] * volatility_multiplier
    lower_bound = -df_labelled['rolling_std'] * volatility_multiplier
    
    # Assign labels
    conditions = [
        (df_labelled['Future_Ret'] > upper_bound),
        (df_labelled['Future_Ret'] < lower_bound)
    ]
    choices = [1, 2] # 1 for Up, 2 for Down, 0 for Flat
    
    df_labelled['label'] = np.select(conditions, choices, default=0)
    
    # Drop NaNs created by shifts and rolling
    df_labelled.dropna(inplace=True)
    
    return df_labelled
