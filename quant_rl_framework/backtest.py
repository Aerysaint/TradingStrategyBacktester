import os
import pandas as pd
import numpy as np
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots
import logging
from config import TradingConfig
from data_processor import MarketDataPipeline
from trading_env import OptionsProxyEnv

try:
    from stable_baselines3 import PPO
except ImportError:
    pass

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, model_path: str, csv_path: str, config: TradingConfig):
        self.config = config
        self.pipeline = MarketDataPipeline(csv_path, config)
        self.df = self.pipeline.build_features()
        self.df = self.pipeline.generate_rolling_scaling(self.df)
        self.env = OptionsProxyEnv(self.df, config)
        
        try:
            self.model = PPO.load(model_path)
        except Exception:
            from sb3_contrib import RecurrentPPO
            self.model = RecurrentPPO.load(model_path)
            
    def run_backtest(self):
        obs, _ = self.env.reset()
        done = False
        truncated = False
        
        self.history = []
        lstm_states = None
        
        while not (done or truncated):
            action, lstm_states = self.model.predict(obs, state=lstm_states, deterministic=True)
            obs, reward, done, truncated, info = self.env.step(action)
            
            if 'balance' not in info:
                break
                
            step_data = {
                'timestamp': self.env.df['timestamp'].iloc[self.env.current_step],
                'close': self.env.df['close'].iloc[self.env.current_step],
                'action': action,
                'balance': info['balance'],
                'fee': info['fee'],
                'position': info['position']
            }
            self.history.append(step_data)
            
        return pd.DataFrame(self.history)
        
    def generate_report(self, history_df: pd.DataFrame):
        net_profit = history_df['balance'].iloc[-1] - self.config.INITIAL_BALANCE
        returns = history_df['balance'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60) if returns.std() != 0 else 0
        
        peak = history_df['balance'].cummax()
        drawdown = (history_df['balance'] - peak) / peak
        max_dd = drawdown.min()
        
        total_fees = history_df['fee'].sum()
        
        # Win rate logic: count positive returns
        winning_steps = len(returns[returns > 0])
        total_active_steps = len(returns[returns != 0])
        win_rate = winning_steps / total_active_steps if total_active_steps > 0 else 0.0
        
        logger.info(f"Net Profit: {net_profit:.2f}")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Max Drawdown: {max_dd:.2%}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Total Brokerage: {total_fees:.2f}")
        
        self._plot_results(history_df)
        
    def _plot_results(self, history_df: pd.DataFrame):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            subplot_titles=('NIFTY 50 Execution', 'Equity Curve'))
        
        df_plot = self.df.iloc[self.config.WINDOW_SIZE:self.config.WINDOW_SIZE + len(history_df)].copy()
        
        # Candlesticks
        fig.add_trace(plotly_go.Candlestick(
            x=df_plot['timestamp'],
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='Price'
        ), row=1, col=1)
        
        # Actions markers
        long_entries = history_df[(history_df['position'] == 1) & (history_df['position'].shift(1).fillna(0) == 0)]
        short_entries = history_df[(history_df['position'] == 2) & (history_df['position'].shift(1).fillna(0) == 0)]
        exits = history_df[(history_df['position'] == 0) & (history_df['position'].shift(1).fillna(0) != 0)]
        
        fig.add_trace(plotly_go.Scatter(
            x=long_entries['timestamp'], y=long_entries['close'],
            mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
            name='Long Entry'
        ), row=1, col=1)
        
        fig.add_trace(plotly_go.Scatter(
            x=short_entries['timestamp'], y=short_entries['close'],
            mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
            name='Short Entry'
        ), row=1, col=1)
        
        fig.add_trace(plotly_go.Scatter(
            x=exits['timestamp'], y=exits['close'],
            mode='markers', marker=dict(symbol='circle', size=10, color='blue'),
            name='Exit'
        ), row=1, col=1)
        
        # Equity Curve
        fig.add_trace(plotly_go.Scatter(
            x=history_df['timestamp'], y=history_df['balance'],
            mode='lines', name='Portfolio Value', line=dict(color='purple')
        ), row=2, col=1)
        
        fig.update_layout(xaxis_rangeslider_visible=False, title='Backtest Results')
        fig.write_html('backtest_results.html')
        logger.info("Saved backtest_results.html")
        
if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__dirname__), "datasets", "NIFTY 50_minute.csv")
    bt = Backtester("ppo_trading_model.zip", dataset_path, TradingConfig())
    res = bt.run_backtest()
    bt.generate_report(res)
