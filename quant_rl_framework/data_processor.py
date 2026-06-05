import pandas as pd
import numpy as np
import ta
import logging
from typing import Tuple
from config import TradingConfig

logger = logging.getLogger(__name__)

class MarketDataPipeline:
    def __init__(self, csv_path: str, config: TradingConfig):
        self.csv_path = csv_path
        self.config = config
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            logger.error(f"Failed to load data from {self.csv_path}: {e}")
            raise
        
        df.columns = [col.lower() for col in df.columns]
        
        if 'date' in df.columns and 'timestamp' not in df.columns:
            df.rename(columns={'date': 'timestamp'}, inplace=True)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
        return df

    def build_features(self) -> pd.DataFrame:
        logger.info("Building technical indicator features...")
        df = self.df.copy()
        
        # Momentum: RSI
        rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi_indicator.rsi()
        
        # Trend: MACD
        macd_indicator = ta.trend.MACD(close=df['close'])
        df['macd'] = macd_indicator.macd()
        
        # Volatility: ATR
        atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr_indicator.average_true_range()
        
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def generate_rolling_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying rolling z-score normalization...")
        scaled_df = df.copy()
        features = ['rsi', 'macd', 'atr']
        epsilon = 1e-8
        
        for feature in features:
            rolling_mean = scaled_df[feature].rolling(window=self.config.WINDOW_SIZE).mean()
            rolling_std = scaled_df[feature].rolling(window=self.config.WINDOW_SIZE).std()
            scaled_df[f'{feature}_scaled'] = (scaled_df[feature] - rolling_mean) / (rolling_std + epsilon)
            
        scaled_df.dropna(inplace=True)
        scaled_df.reset_index(drop=True, inplace=True)
        return scaled_df
