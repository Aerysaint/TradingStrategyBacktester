import pytest
import pandas as pd
import numpy as np
import os
from quant_rl_framework.config import TradingConfig
from quant_rl_framework.data_processor import MarketDataPipeline

def test_data_isolation(tmp_path):
    # Create dummy data
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=100, freq="1min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.randn(100).cumsum() + 100,
        "high": np.random.randn(100).cumsum() + 105,
        "low": np.random.randn(100).cumsum() + 95,
        "close": np.random.randn(100).cumsum() + 100,
        "volume": np.random.randint(100, 1000, size=100)
    })
    
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)
    
    config = TradingConfig(WINDOW_SIZE=10)
    pipeline = MarketDataPipeline(str(csv_path), config)
    
    features_df = pipeline.build_features()
    scaled_df = pipeline.generate_rolling_scaling(features_df)
    
    # Assert data isolation: the scaled value at time t should only depend on [t-WINDOW_SIZE+1 : t]
    # We test this by slightly modifying t+1 and ensuring t remains identical.
    
    df_leak = df.copy()
    df_leak.loc[50, "close"] = 999999.0
    csv_leak_path = tmp_path / "dummy_leak.csv"
    df_leak.to_csv(csv_leak_path, index=False)
    
    pipeline_leak = MarketDataPipeline(str(csv_leak_path), config)
    features_leak = pipeline_leak.build_features()
    scaled_leak = pipeline_leak.generate_rolling_scaling(features_leak)
    
    # Row 49 in the original time series should be completely unaffected by the massive change in Row 50
    # Note: Drops of NA values might shift indices, so let's match by timestamp
    
    t_49 = df.loc[49, "timestamp"]
    
    row_clean = scaled_df[scaled_df["timestamp"] == t_49].iloc[0]
    row_leak = scaled_leak[scaled_leak["timestamp"] == t_49].iloc[0]
    
    assert np.isclose(row_clean["rsi_scaled"], row_leak["rsi_scaled"]), "Lookahead bias detected in RSI scaling"
    assert np.isclose(row_clean["macd_scaled"], row_leak["macd_scaled"]), "Lookahead bias detected in MACD scaling"
