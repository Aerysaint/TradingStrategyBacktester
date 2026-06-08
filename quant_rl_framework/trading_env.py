import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any
from config import TradingConfig

logger = logging.getLogger(__name__)

class OptionsProxyEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, config: TradingConfig):
        super(OptionsProxyEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.config = config
        
        # 0: Flat, 1: Long, 2: Short
        self.action_space = spaces.Discrete(3)
        
        # Feature columns: rsi_scaled, macd_scaled, atr_scaled
        self.feature_cols = [col for col in self.df.columns if col.endswith('_scaled')]
        self.num_features = len(self.feature_cols)
        
        # Observation Box: Window Size rows x (Features + Position Status + Minutes in Psn)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.config.WINDOW_SIZE, self.num_features + 2), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = self.config.INITIAL_BALANCE
        self.position = 0
        self.minutes_in_position = 0
        self.entry_price = 0.0
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.balance = self.config.INITIAL_BALANCE
        self.position = 0
        self.minutes_in_position = 0
        self.entry_price = 0.0
        self.current_step = self.config.WINDOW_SIZE
        
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        start_idx = self.current_step - self.config.WINDOW_SIZE
        end_idx = self.current_step
        
        obs_features = self.df[self.feature_cols].iloc[start_idx:end_idx].values
        
        # Add position and minutes to obs
        pos_array = np.full((self.config.WINDOW_SIZE, 1), self.position, dtype=np.float32)
        min_psn_array = np.full((self.config.WINDOW_SIZE, 1), self.minutes_in_position, dtype=np.float32)
        
        obs = np.hstack((obs_features, pos_array, min_psn_array))
        return obs.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        reward = 0.0
        done = False
        truncated = False
        
        if self.current_step >= len(self.df) - self.config.LATENCY_MINUTES - 1:
            truncated = True
            return self._get_observation(), reward, done, truncated, {}
            
        current_price = self.df['close'].iloc[self.current_step]
        prev_price = self.df['close'].iloc[self.current_step - 1]
        
        # Execution price with latency and slippage
        exec_idx = self.current_step + self.config.LATENCY_MINUTES
        market_exec_price = self.df['close'].iloc[exec_idx]
        
        slippage_ticks = self.np_random.uniform(0, self.config.SLIPPAGE_MAX_TICKS)
        slippage_amt = slippage_ticks * 0.05
        
        step_reward = 0.0
        fee = 0.0
        
        # PnL accounting for existing position
        if self.position == 1:
            step_reward += (current_price - prev_price)
        elif self.position == 2:
            step_reward += (prev_price - current_price)
            
        # State transitions
        just_entered = False
        trade_executed = False
        new_position = self.position

        # Action 0: Hold
        # Action 1: Buy
        # Action 2: Sell
        if action == 1:
            if self.position == 0:
                new_position = 1 # Enter Long
                trade_executed = True
                just_entered = True
            elif self.position == 2:
                new_position = 0 # Exit Short
                trade_executed = True
        elif action == 2:
            if self.position == 0:
                new_position = 2 # Enter Short
                trade_executed = True
                just_entered = True
            elif self.position == 1:
                new_position = 0 # Exit Long
                trade_executed = True

        if trade_executed:
            if action == 1: # Buying
                p_exec = market_exec_price + slippage_amt
            else: # Selling
                p_exec = market_exec_price - slippage_amt
                
            fee += p_exec * self.config.TRANSACTION_FEE_PCT
            
            if just_entered:
                self.entry_price = p_exec
                self.minutes_in_position = 0
            
            self.position = new_position
            
        # Holding penalty
        penalty = 0.0
        if self.position != 0:
            if not just_entered:
                self.minutes_in_position += 1
            # Linear theta decay: constant penalty per minute held
            penalty = self.config.THETA_DECAY_COEFF
            
        total_step_reward = step_reward - fee - penalty
        self.balance += total_step_reward
        
        # Ruin condition
        if self.balance < self.config.INITIAL_BALANCE * self.config.RUIN_THRESHOLD:
            done = True
            
        info = {
            'balance': self.balance,
            'position': self.position,
            'fee': fee,
            'penalty': penalty,
            'step_reward': total_step_reward
        }
        
        return self._get_observation(), total_step_reward, done, truncated, info
