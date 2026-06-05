import pytest
import pandas as pd
import numpy as np
from quant_rl_framework.config import TradingConfig
from quant_rl_framework.trading_env import OptionsProxyEnv

def test_theta_penalty():
    # Construct completely static price data for 50 periods
    dates = pd.date_range("2026-01-01", periods=50, freq="1min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.full(50, 100.0),
        "high": np.full(50, 100.0),
        "low": np.full(50, 100.0),
        "close": np.full(50, 100.0),
        "volume": np.full(50, 100)
    })
    
    # Add dummy scaled features to satisfy the environment observation space
    df['rsi_scaled'] = 0.0
    df['macd_scaled'] = 0.0
    df['atr_scaled'] = 0.0
    
    # Disable slippage and fees for clear theta testing
    config = TradingConfig(
        WINDOW_SIZE=5, 
        SLIPPAGE_MAX_TICKS=0, 
        TRANSACTION_FEE_PCT=0.0,
        THETA_DECAY_COEFF=0.01,
        LATENCY_MINUTES=0
    )
    
    env = OptionsProxyEnv(df, config)
    env.reset()
    
    # 0 action (Flat) should yield 0 reward
    _, r_flat, _, _, _ = env.step(0)
    assert r_flat == 0.0
    
    # Force Long Entry (1)
    _, r_entry, _, _, _ = env.step(1)
    # At entry, minute_in_position=0, penalty=0
    
    # Step through holding for 10 periods
    rewards = []
    for _ in range(10):
        _, r, _, _, _ = env.step(1)
        rewards.append(r)
        
    # Verify that the reward strictly decreases due to quadratic penalty
    for i in range(1, len(rewards)):
        # Rewards are negative (penalties), decreasing means becoming more negative
        assert rewards[i] < rewards[i-1], f"Reward did not strictly decrease quadratically: step {i}"
        
    # Mathematically verify penalty: penalty_t = k * t^2
    # At index 2 (t=3 since entry), penalty should be 0.01 * 9 = -0.09
    assert np.isclose(rewards[2], -0.01 * (3 ** 2)), f"Expected penalty {-0.01 * 9}, got {rewards[2]}"
