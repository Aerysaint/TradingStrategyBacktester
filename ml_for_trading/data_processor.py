import pandas as pd
import numpy as np
import os

def load_and_fuse_data(data_dir, norm_window, atr_window):
    """
    Loads 1m, 5m, 15m, 60m, 1d datasets and fuses them to the 5m index.
    """
    df_1m = pd.read_csv(os.path.join(data_dir, "NIFTY 50_minute.csv"))
    df_5m = pd.read_csv(os.path.join(data_dir, "NIFTY 50_5minute.csv"))
    df_15m = pd.read_csv(os.path.join(data_dir, "NIFTY 50_15minute.csv"))
    df_60m = pd.read_csv(os.path.join(data_dir, "NIFTY 50_60minute.csv"))
    df_1d = pd.read_csv(os.path.join(data_dir, "NIFTY 50_day.csv"))

    def prep_df(df, prefix):
        df = df.copy()
        if 'date' in df.columns:
            df.rename(columns={'date': 'datetime'}, inplace=True)
        if 'datetime' not in df.columns:
            df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
        if prefix:
            df = df.add_suffix(f'_{prefix}')
        return df

    df_1m = prep_df(df_1m, '1m')
    df_5m = prep_df(df_5m, '')
    
    # CRITICAL FIX for Data Leakage: Higher timeframe timestamps typically represent 
    # the start of the bar. To prevent the 09:15 5-min candle from seeing the 
    # 09:15-09:30 15-min closure, we explicitly shift the timeframe down by 1 completely 
    # sealing the "future" out of the present features.
    df_15m = prep_df(df_15m, '15m').shift(1)
    df_60m = prep_df(df_60m, '60m').shift(1)
    df_1d = prep_df(df_1d, '1d').shift(1)

    # Aggregate 1m to 5m (e.g., rolling 5-period momentum of close)
    df_1m_rolled = df_1m[['close_1m']].rolling(window=5).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1, raw=False)
    df_1m_rolled.rename(columns={'close_1m': 'mom_1m_5p'}, inplace=True)
    df_1m_resampled = df_1m_rolled.reindex(df_5m.index, method='ffill')

    # Forward fill higher timeframes
    df_15m_aligned = pd.merge_asof(df_5m[[]], df_15m, left_index=True, right_index=True, direction='backward')
    df_60m_aligned = pd.merge_asof(df_5m[[]], df_60m, left_index=True, right_index=True, direction='backward')
    df_1d_aligned = pd.merge_asof(df_5m[[]], df_1d, left_index=True, right_index=True, direction='backward')

    # Merge all
    df_fused = df_5m.join([df_1m_resampled, df_15m_aligned, df_60m_aligned, df_1d_aligned], how='left')

    # Cyclical Time Encoding
    df_fused['minute_of_day'] = df_fused.index.hour * 60 + df_fused.index.minute
    df_fused['day_of_week'] = df_fused.index.dayofweek

    df_fused['sin_time'] = np.sin(2 * np.pi * df_fused['minute_of_day'] / (24 * 60))
    df_fused['cos_time'] = np.cos(2 * np.pi * df_fused['minute_of_day'] / (24 * 60))
    df_fused['sin_day'] = np.sin(2 * np.pi * df_fused['day_of_week'] / 5)
    df_fused['cos_day'] = np.cos(2 * np.pi * df_fused['day_of_week'] / 5)
    df_fused.drop(columns=['minute_of_day', 'day_of_week'], inplace=True)

    # Dynamic Indicators on 5m
    def calculate_atr(df, window):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window).mean()

    df_fused['SMA_5m'] = df_fused['close'].rolling(window=14).mean()
    df_fused['EMA_5m'] = df_fused['close'].ewm(span=14, adjust=False).mean()
    exp1 = df_fused['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_fused['close'].ewm(span=26, adjust=False).mean()
    df_fused['MACD'] = exp1 - exp2
    df_fused['ATR'] = calculate_atr(df_fused, atr_window)
    
    # NEW FEATURES: Momentum & Volatility
    df_fused['ROC_3'] = df_fused['close'].pct_change(periods=3)
    df_fused['ROC_5'] = df_fused['close'].pct_change(periods=5)
    df_fused['ROC_10'] = df_fused['close'].pct_change(periods=10)
    
    df_fused['date_only'] = df_fused.index.date
    if 'volume' in df_fused.columns:
        # Distance from VWAP (VWAP resets daily)
        vwap_numer = df_fused.groupby('date_only').apply(lambda x: (x['close'] * x['volume']).cumsum()).reset_index(level=0, drop=True)
        vwap_denom = df_fused.groupby('date_only')['volume'].cumsum()
        vwap = vwap_numer / (vwap_denom + 1e-8)
        df_fused['Dist_VWAP'] = (df_fused['close'] - vwap) / (vwap + 1e-8)
    else:
        df_fused['Dist_VWAP'] = 0
        
    df_fused['HL_Spread'] = (df_fused['high'] - df_fused['low']).rolling(window=5).mean()

    # Store unnormalized price and ATR for realistic PnL tracking in RL environment
    df_fused['close_unnorm'] = df_fused['close']
    df_fused['ATR_unnorm'] = df_fused['ATR']

    # Dynamic Normalization (Z-score grouped by date to avoid overnight look-ahead)
    def normalize_group(group, window):
        mean = group.rolling(window=window, min_periods=1).mean()
        std = group.rolling(window=window, min_periods=1).std()
        std[std == 0] = 1e-8
        return (group - mean) / std

    cols_to_norm = [c for c in df_fused.columns if c not in ['sin_time', 'cos_time', 'sin_day', 'cos_day', 'close_unnorm', 'ATR_unnorm'] and ('close' in c or 'open' in c or 'high' in c or 'low' in c or 'volume' in c or 'mom' in c or 'SMA' in c or 'EMA' in c or 'MACD' in c or 'ATR' in c or 'ROC' in c or 'Dist_VWAP' in c or 'HL_Spread' in c)]
    
    df_fused['date_only'] = df_fused.index.date
    # Ensure cols_to_norm actually exist, sometimes index issues might occur.
    df_fused[cols_to_norm] = df_fused.groupby('date_only')[cols_to_norm].transform(lambda x: normalize_group(x, norm_window))
    df_fused.drop(columns=['date_only'], inplace=True)

    df_fused.replace([np.inf, -np.inf], 0, inplace=True)
    df_fused.dropna(inplace=True)
    return df_fused
