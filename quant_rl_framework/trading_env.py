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
        
        # Pre-allocate numpy arrays for 100x speedup over pandas .iloc
        self._features_arr = self.df[self.feature_cols].values
        self._close_arr = self.df['close'].values
        if 'timestamp' in self.df.columns:
            self._is_new_day_arr = (self.df['timestamp'].dt.date != self.df['timestamp'].shift(1).dt.date).values
        else:
            self._is_new_day_arr = np.zeros(len(self.df), dtype=bool)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.balance = self.config.INITIAL_BALANCE
        self.position = 0
        self.minutes_in_position = 0
        self.entry_price = 0.0
        
        max_start = len(self.df) - self.config.MAX_STEPS_PER_EPISODE - 1
        if max_start > self.config.WINDOW_SIZE:
            self.current_step = self.np_random.integers(self.config.WINDOW_SIZE, max_start)
        else:
            self.current_step = self.config.WINDOW_SIZE
            
        self.start_step = self.current_step
        
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        start_idx = self.current_step - self.config.WINDOW_SIZE
        end_idx = self.current_step
        
        obs_features = self._features_arr[start_idx:end_idx]
        
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
        
        if self.current_step >= len(self.df) - 1 or (self.current_step - self.start_step) >= self.config.MAX_STEPS_PER_EPISODE:
            truncated = True
            return self._get_observation(), reward, done, truncated, {}
            
        current_price = self._close_arr[self.current_step]
        prev_price = self._close_arr[self.current_step - 1]
        
        slippage_ticks = self.np_random.uniform(0, self.config.SLIPPAGE_MAX_TICKS)
        slippage_amt = slippage_ticks * 0.05
        
        step_reward = 0.0
        fee = 0.0
        
        new_position = self.position
        just_entered = False

        # Action 0: Hold
        # Action 1: Buy
        # Action 2: Sell
        if action == 1:
            if self.position == 0:
                new_position = 1 # Enter Long
            elif self.position == 2:
                new_position = 0 # Exit Short
        elif action == 2:
            if self.position == 0:
                new_position = 2 # Enter Short
            elif self.position == 1:
                new_position = 0 # Exit Long

        # Handle Exits
        if self.position == 1 and new_position == 0:
            p_exec = current_price - slippage_amt
            fee += p_exec * self.config.TRANSACTION_FEE_PCT
            step_reward += (p_exec - prev_price)
        elif self.position == 2 and new_position == 0:
            p_exec = current_price + slippage_amt
            fee += p_exec * self.config.TRANSACTION_FEE_PCT
            step_reward += (prev_price - p_exec)
            
        # Handle Holds
        elif self.position == 1 and new_position == 1:
            step_reward += (current_price - prev_price)
        elif self.position == 2 and new_position == 2:
            step_reward += (prev_price - current_price)
            
        # Handle Entries
        elif self.position == 0 and new_position == 1:
            p_exec = current_price + slippage_amt
            fee += p_exec * self.config.TRANSACTION_FEE_PCT
            step_reward += (current_price - p_exec)
            self.entry_price = p_exec
            self.minutes_in_position = 0
            just_entered = True
        elif self.position == 0 and new_position == 2:
            p_exec = current_price - slippage_amt
            fee += p_exec * self.config.TRANSACTION_FEE_PCT
            step_reward += (p_exec - current_price)
            self.entry_price = p_exec
            self.minutes_in_position = 0
            just_entered = True

        self.position = new_position
            
        # Holding penalty
        penalty = 0.0
        if self.position != 0:
            if not just_entered:
                self.minutes_in_position += 1
            # Increasing penalty as you hold longer (linear increase makes total cost quadratic)
            penalty += self.config.THETA_DECAY_COEFF * (1.0 + (self.minutes_in_position / 60.0))
            
            # Overnight penalty
            if self._is_new_day_arr[self.current_step]:
                penalty += self.config.OVERNIGHT_HOLD_PENALTY
            
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
