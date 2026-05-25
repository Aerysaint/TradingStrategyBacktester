import pandas as pd
import numpy as np
import pandas_ta as ta
import scipy.stats as si
from datetime import datetime, timedelta, time
from .data import get_ohlcv_data

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0: return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0: return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

def get_next_expiry(dt: datetime) -> datetime:
    change_date = datetime(2025, 9, 1)
    target_weekday = 3 if dt < change_date else 1
    days_ahead = target_weekday - dt.weekday()
    if days_ahead < 0:
        days_ahead += 7
    elif days_ahead == 0:
        if dt.time() >= time(15, 30):
            days_ahead += 7
    expiry_date = dt + timedelta(days=days_ahead)
    return datetime.combine(expiry_date.date(), time(15, 30))

def run_backtest(df: pd.DataFrame, indicators: list, rules: list, base_timeframe: str = "1minute", options_config: dict = {}):
    if df.empty:
        return {"error": "Not enough data"}

    df = df.copy()
    
    # Fix visual glitches by removing any duplicate timestamps before processing
    df = df.drop_duplicates(subset=['date'])
    
    # Evaluate indicators dynamically
    indicator_series = {}
    indicator_data_cache = {}
    
    # Convert base df date to datetime and set as index for safe merging
    df['date_dt'] = pd.to_datetime(df['date'])
    df_indexed = df.set_index('date_dt')
    
    start_dt = df_indexed.index.min().strftime('%Y-%m-%d %H:%M:%S')
    end_dt = df_indexed.index.max().strftime('%Y-%m-%d %H:%M:%S')

    # Figure out required indicator timeframes from rules
    required_evals = set()
    ind_map = { (ind.id if hasattr(ind, 'id') else f"{ind.name.lower()}_{ind.id}"): ind for ind in indicators }
    
    # Always evaluate on base timeframe for charting
    for ind_id in ind_map.keys():
        required_evals.add((ind_id, base_timeframe))
        
    for rule in rules:
        tf = getattr(rule, 'timeframe', None) or base_timeframe
        for cond in rule.conditions:
            left_id = cond.left.split('.')[0]
            right_id = str(cond.right).split('.')[0]
            if left_id in ind_map:
                required_evals.add((left_id, tf))
            if right_id in ind_map:
                required_evals.add((right_id, tf))
                    
    # Evaluate indicators for each required timeframe
    df_evals = {} # (ind_id, tf): pd.Series or pd.DataFrame
    
    for ind_id, tf in required_evals:
        ind = ind_map[ind_id]
        name = ind.name.lower()
        kwargs = ind.params
        
        target_df = df_indexed
        if tf != base_timeframe:
            if tf not in indicator_data_cache:
                tf_df = get_ohlcv_data(tf, start_dt, end_dt, limit=200000)
                if not tf_df.empty:
                    tf_df['date_dt'] = pd.to_datetime(tf_df['date'])
                    tf_df.set_index('date_dt', inplace=True)
                    indicator_data_cache[tf] = tf_df
                else:
                    indicator_data_cache[tf] = df_indexed # fallback
            target_df = indicator_data_cache[tf]

        res = target_df.ta(kind=name, append=False, **kwargs)
        
        # Merge back to base timeframe
        if tf != base_timeframe:
            if isinstance(res, pd.Series):
                res = res.reindex(df_indexed.index, method='ffill')
            elif isinstance(res, pd.DataFrame):
                res = res.reindex(df_indexed.index, method='ffill')
                
        df_evals[(ind_id, tf)] = res
        
        # Only output base_timeframe for charting to avoid chart clutter
        if tf == base_timeframe:
            if isinstance(res, pd.Series):
                df[ind_id] = res.values
                indicator_series[ind_id] = pd.Series(res.values).astype(object).replace([np.nan, np.inf, -np.inf], None).to_dict()
            elif isinstance(res, pd.DataFrame):
                for col in res.columns:
                    base_name = col.split('_')[0].lower()
                    full_col_id = f"{ind_id}.{base_name}"
                    df[full_col_id] = res[col].values
                    indicator_series[full_col_id] = pd.Series(res[col].values).astype(object).replace([np.nan, np.inf, -np.inf], None).to_dict()

    # Evaluate rules
    df['signal'] = 0
    df['buy_cond'] = False
    df['sell_cond'] = False

    for rule in rules:
        cond_series = pd.Series(True, index=df.index)
        tf = getattr(rule, 'timeframe', None) or base_timeframe
        
        for cond in rule.conditions:
            def get_series(col_str):
                base_id = str(col_str).split('.')[0]
                if any(ind_id == base_id for (ind_id, _), _ in df_evals.items()):
                    res = df_evals.get((base_id, tf))
                    if res is None: res = df_evals.get((base_id, base_timeframe))
                    if isinstance(res, pd.DataFrame):
                        if '.' in str(col_str):
                            sub_name = str(col_str).split('.')[1]
                            for col in res.columns:
                                if col.split('_')[0].lower() == sub_name:
                                    return res[col].values
                        return res.iloc[:, 0].values
                    else:
                        return res.values
                else:
                    if str(col_str) in df.columns:
                        return df[col_str].values
                    else:
                        return float(col_str)
                        
            left = get_series(cond.left)
            right = get_series(cond.right)
            
            left_s = pd.Series(left, index=df.index) if isinstance(left, np.ndarray) else left
            right_s = pd.Series(right, index=df.index) if isinstance(right, np.ndarray) else right
            
            if cond.operator == '>':
                cond_series &= (left_s > right_s)
            elif cond.operator == '<':
                cond_series &= (left_s < right_s)
            elif cond.operator == '>=':
                cond_series &= (left_s >= right_s)
            elif cond.operator == '<=':
                cond_series &= (left_s <= right_s)
            elif cond.operator == '==':
                cond_series &= (left_s == right_s)
            elif cond.operator == 'crossover':
                cond_series &= (left_s > right_s) & (left_s.shift(1) <= right_s.shift(1))
            elif cond.operator == 'crossunder':
                cond_series &= (left_s < right_s) & (left_s.shift(1) >= right_s.shift(1))

        if rule.action == 'buy':
            df['buy_cond'] |= cond_series
        elif rule.action == 'sell':
            df['sell_cond'] |= cond_series

    df.loc[df['buy_cond'], 'signal'] = 1
    df.loc[df['sell_cond'], 'signal'] = -1
    
    df['position'] = df['signal'].shift(1).fillna(0)
    # Simple holding: 1 = long, -1 = short
    # Here, we need to hold position until opposite signal
    positions = []
    current_pos = 0
    for s in df['signal']:
        if s == 1:
            current_pos = 1
        elif s == -1:
            current_pos = -1
        positions.append(current_pos)
    df['position'] = np.array(positions).flatten()
    # Actually wait, shift by 1 to make decisions after candle close
    df['position'] = pd.Series(positions).shift(1).fillna(0)

    df['trade'] = df['position'].diff()
    
    capital = 100000
    entry_price = 0
    
    BROKERAGE = 20.0
    EXCHANGE_FEE = 0.0000325
    STT = 0.00025
    
    pnl = 0
    max_drawdown = 0
    peak_capital = capital
    winning_trades = 0
    losing_trades = 0
    trade_list = []
    
    prices = df['close'].values
    positions_arr = df['position'].values
    dates = df['date'].values
    date_dts = pd.to_datetime(df['date']).values
    
    # Options Config
    opt_enabled = options_config.get('enabled', False)
    opt_iv = options_config.get('iv', 0.15)
    opt_r = options_config.get('risk_free_rate', 0.05)
    
    entry_strike = 0
    entry_opt_price = 0
    entry_time = None
    expiry_time = None
    trade_qty = 0
    
    for i in range(1, len(df)):
        pos_prev = positions_arr[i-1]
        pos_curr = positions_arr[i]
        price = prices[i]
        date = dates[i]
        date_dt = date_dts[i]
        
        # Options Expiration Check
        if opt_enabled and pos_prev != 0 and expiry_time is not None:
            if date_dt >= expiry_time:
                pos_curr = 0 # Force close at expiration
                positions_arr[i] = 0 # update array so next iteration knows we are flat
        
        if pos_curr != pos_prev:
            if pos_prev == 1:
                # Sell to close long (Call if options, Equity if not)
                if opt_enabled:
                    t_exit = max(0.0, (expiry_time - date_dt).total_seconds() / (365.0 * 86400))
                    exit_price = black_scholes_call(price, entry_strike, t_exit, opt_r, opt_iv)
                    trade_val = trade_qty * exit_price
                    total_cost = BROKERAGE + (trade_val * STT) + (trade_val * EXCHANGE_FEE)
                    trade_pnl = (exit_price - entry_opt_price) * trade_qty - total_cost
                else:
                    trade_val = trade_qty * price
                    total_cost = BROKERAGE + (trade_val * STT) + (trade_val * EXCHANGE_FEE)
                    trade_pnl = (price - entry_price) * trade_qty - total_cost
                    
                pnl += trade_pnl
                capital += trade_pnl
                if trade_pnl > 0: winning_trades += 1
                else: losing_trades += 1
                trade_list.append({"time": date, "action": "sell", "price": float(price), "pnl": float(trade_pnl)})
                
            elif pos_prev == -1:
                # Buy to close short (Put if options, Equity short if not)
                if opt_enabled:
                    t_exit = max(0.0, (expiry_time - date_dt).total_seconds() / (365.0 * 86400))
                    exit_price = black_scholes_put(price, entry_strike, t_exit, opt_r, opt_iv)
                    trade_val = trade_qty * exit_price
                    total_cost = BROKERAGE + (trade_val * EXCHANGE_FEE)
                    trade_pnl = (exit_price - entry_opt_price) * trade_qty - total_cost
                else:
                    trade_val = trade_qty * price
                    total_cost = BROKERAGE + (trade_val * EXCHANGE_FEE)
                    trade_pnl = (entry_price - price) * trade_qty - total_cost
                    
                pnl += trade_pnl
                capital += trade_pnl
                if trade_pnl > 0: winning_trades += 1
                else: losing_trades += 1
                trade_list.append({"time": date, "action": "buy", "price": float(price), "pnl": float(trade_pnl)})
                
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (peak_capital - capital) / peak_capital
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
            if pos_curr == 1:
                # Enter Long
                if opt_enabled:
                    entry_strike = price
                    entry_time = date_dt
                    expiry_time = get_next_expiry(entry_time)
                    t_entry = (expiry_time - entry_time).total_seconds() / (365.0 * 86400)
                    entry_opt_price = black_scholes_call(price, entry_strike, t_entry, opt_r, opt_iv)
                    trade_qty = capital // entry_opt_price if entry_opt_price > 0 else 0
                    trade_val = trade_qty * entry_opt_price
                    capital -= (BROKERAGE + (trade_val * EXCHANGE_FEE))
                else:
                    trade_qty = capital // price if price > 0 else 0
                    trade_val = trade_qty * price
                    capital -= (BROKERAGE + (trade_val * EXCHANGE_FEE))
                    entry_price = price
                    
                trade_list.append({"time": date, "action": "buy", "price": float(price), "pnl": 0.0})
                
            elif pos_curr == -1:
                # Enter Short
                if opt_enabled:
                    entry_strike = price
                    entry_time = date_dt
                    expiry_time = get_next_expiry(entry_time)
                    t_entry = (expiry_time - entry_time).total_seconds() / (365.0 * 86400)
                    entry_opt_price = black_scholes_put(price, entry_strike, t_entry, opt_r, opt_iv)
                    trade_qty = capital // entry_opt_price if entry_opt_price > 0 else 0
                    trade_val = trade_qty * entry_opt_price
                    capital -= (BROKERAGE + (trade_val * STT) + (trade_val * EXCHANGE_FEE))
                else:
                    trade_qty = capital // price if price > 0 else 0
                    trade_val = trade_qty * price
                    capital -= (BROKERAGE + (trade_val * STT) + (trade_val * EXCHANGE_FEE))
                    entry_price = price
                    
                trade_list.append({"time": date, "action": "sell", "price": float(price), "pnl": 0.0})

    total_trades = winning_trades + losing_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    sharpe_ratio = (pnl / total_trades) / (df["close"].std() + 1e-9) if total_trades > 0 else 0
    
    # Format indicator_series array specifically for frontend mapping.
    # We will build an array of lines: [{"id": "ind1", "data": [{"time": ..., "value": ...}, ...]}...]
    formatted_indicators = []
    
    dates_list = df['date'].tolist()
    for col_id, s in indicator_series.items():
        data_points = []
        for i, val in enumerate(s.values()):
            if val is not None and not np.isnan(val):
                data_points.append({"time": dates_list[i], "value": float(val)})
        formatted_indicators.append({
            "id": col_id,
            "data": data_points
        })

    return {
        "metrics": {
            "net_pnl": float(pnl),
            "max_drawdown_pct": float(max_drawdown * 100),
            "win_rate": float(win_rate * 100),
            "total_trades": int(total_trades),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "sharpe_ratio": float(sharpe_ratio)
        },
        "trades": trade_list,
        "indicators": formatted_indicators
    }
