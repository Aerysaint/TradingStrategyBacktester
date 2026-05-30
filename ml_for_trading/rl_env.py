import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df, seq_length, stop_loss_multiplier, transaction_cost=2.5, total_training_steps=50000):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.seq_length = seq_length
        self.stop_loss_multiplier = stop_loss_multiplier
        self.max_transaction_cost = transaction_cost # Absolute Index Points penalty per side
        self.total_training_steps = total_training_steps
        self.global_step = 0
        
        # Assuming df has 'close_unnorm' for actual PnL tracking 
        # and 'ATR_unnorm' for dynamic SL in actual price points.
        # If not, we fall back to 'close' and 'ATR'.
        price_col = 'close_unnorm' if 'close_unnorm' in self.df.columns else 'close'
        atr_col = 'ATR_unnorm' if 'ATR_unnorm' in self.df.columns else 'ATR'
        
        self.prices = self.df[price_col].values
        self.atrs = self.df[atr_col].values
        
        self.feature_cols = [c for c in self.df.columns if c not in ['label', 'Future_Ret', 'rolling_std', price_col, atr_col]]
        self.features = self.df[self.feature_cols].values
        
        # State space: flattened sequence of features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.seq_length * len(self.feature_cols),), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3) # 0: Hold/Flat, 1: Buy/Long, 2: Sell/Short
        
        self.current_step = self.seq_length
        self.position = 0 # 1 for Long, -1 for Short, 0 for Flat
        self.entry_price = 0
        self.flat_duration = 0
        self.done = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.seq_length
        self.position = 0
        self.entry_price = 0
        self.flat_duration = 0
        self.done = False
        return self._next_observation(), {}
        
    def _next_observation(self):
        obs = self.features[self.current_step - self.seq_length : self.current_step].flatten()
        return obs.astype(np.float32)
        
    def step(self, action):
        if self.done:
            return self._next_observation(), 0, self.done, False, {}
            
        self.global_step += 1
        
        # Curriculum Learning: Scale transaction costs dynamically from 0 to full fee at 50% training
        current_fee = self.max_transaction_cost * min(1.0, self.global_step / max(1, self.total_training_steps * 0.5))
            
        current_price = self.prices[self.current_step]
        current_atr = self.atrs[self.current_step]
        step_reward = 0
        
        # Target Position Action Space: 0 -> Target Flat, 1 -> Target Long (+1), 2 -> Target Short (-1)
        target_position = 0 if action == 0 else (1 if action == 1 else -1)
        
        # Dynamic Risk Management: Stop Loss
        forced_close = False
        actual_pnl = 0
        
        if self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
            stop_loss_threshold = self.stop_loss_multiplier * current_atr
            
            if unrealized_pnl < -stop_loss_threshold: # Hit stop loss
                forced_close = True
                actual_pnl += unrealized_pnl - current_fee
                step_reward += unrealized_pnl - current_fee
                self.position = 0
                self.entry_price = 0
                
        # Execute orders and Mark-to-Market
        if not forced_close:
            # 1. Execute Position Change (Pay transaction fee if moving)
            if target_position != self.position:
                # If closing an existing position, we pay a fee.
                if self.position != 0:
                    actual_pnl -= current_fee
                    step_reward -= current_fee
                
                # If opening a new position, we pay a fee.
                if target_position != 0:
                    actual_pnl -= current_fee
                    step_reward -= current_fee
                    self.entry_price = current_price
                    
                self.position = target_position
                
            # 2. Mark-to-Market Step Reward (per step PnL change)
            price_change = self.prices[self.current_step] - self.prices[self.current_step - 1]
            if self.position == 1:
                step_reward += price_change
                actual_pnl += price_change
            elif self.position == -1:
                step_reward -= price_change
                actual_pnl -= price_change
        
        # Flat penalty to avoid 100% hold convergence
        if self.position == 0:
            self.flat_duration += 1
        else:
            self.flat_duration = 0
            
        if self.flat_duration > 50:
            step_reward -= 0.1
                
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
            
        # Reward Normalization & Clipping
        step_reward = step_reward / (current_atr + 1e-8)
        step_reward = float(np.clip(step_reward, -15.0, 15.0))
            
        info = {'actual_pnl': float(actual_pnl)}
        return self._next_observation(), step_reward, self.done, False, info
