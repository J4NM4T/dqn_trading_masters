import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta

def _as_1d_series(x, index=None) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        x = x.squeeze("columns")
    arr = np.asarray(x).squeeze()
    return pd.Series(arr, index=index if index is not None else getattr(x, "index", None), dtype=float)

def load_data(ticker:str, start_dte:str, end_dte:str) -> pd.DataFrame:
    ticker = yf.Ticker(ticker)
    historical_data = ticker.history(interval="1d", start=start_dte, end=end_dte)

    df = historical_data.copy()
    df.index = df.index.tz_localize(None)

    close = _as_1d_series(df['Close'], df.index)
    high = _as_1d_series(df['High'], df.index)
    low  = _as_1d_series(df['Low'], df.index)

    # ---------------------------
    # 1. RSI + MACD + Bollinger
    # ---------------------------
    df['feature_rsi_14'] = ta.RSI(close, timeperiod=14)
    macd_line, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['feature_macd'] = macd_line
    df['feature_macd_signal'] = macd_signal
    df['feature_macd_hist'] = macd_hist

    bb_up, bb_mid, bb_low = ta.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    df['feature_bb_low'] = _as_1d_series(bb_low, df.index)
    df['feature_bb_mid'] = _as_1d_series(bb_mid, df.index)
    df['feature_bb_up']  = _as_1d_series(bb_up,  df.index)
    df['feature_close_bb_pos'] = ((close - df['feature_bb_mid']) / (df['feature_bb_up'] - df['feature_bb_low'] + 1e-12)).astype(float)

    # ---------------------------
    # 2. Returns & Volatility
    # ---------------------------
    df['feature_ret_1d'] = close.pct_change()
    df['feature_ret_std_30'] = df['feature_ret_1d'].rolling(window=30, min_periods=30).std()

    # ---------------------------
    # 3. Existing trend (SMA)
    # ---------------------------
    df['feature_sma_5'] = close.rolling(window=5, min_periods=5).mean()
    df['feature_sma_20'] = close.rolling(window=20, min_periods=20).mean()
    df['feature_sma_5_over_20'] = (df['feature_sma_5'] / df['feature_sma_20']) - 1.0

    # ---------------------------
    # 4. NEW TREND FEATURES
    # ---------------------------

    # (A) Momentum – ROC 20d
    df['feature_roc_20'] = close.pct_change(20)

    # (B) Exponential Moving Averages
    df['feature_ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['feature_ema_50'] = close.ewm(span=50, adjust=False).mean()
    df['feature_ema_20_over_50'] = (df['feature_ema_20'] / df['feature_ema_50']) - 1.0

    # (C) ADX – Trend Strength
    df['feature_adx_14'] = ta.ADX(high, low, close, timeperiod=14)

    # (D) ATR% – volatility breakout / trend confirmation
    df['feature_atr_14'] = ta.ATR(high, low, close, timeperiod=14)
    df['feature_atr_pct'] = df['feature_atr_14'] / close

    # (E) Rolling Sharpe 20d (mini trend quality) 
    rolling_mean = df['feature_ret_1d'].rolling(20).mean() 
    rolling_std = df['feature_ret_1d'].rolling(20).std() 
    df['feature_sharpe_20'] = rolling_mean - 0.02 / (rolling_std + 1e-12)

    # ---------------------------
    # 5. Scaling
    # ---------------------------
    df = df.dropna().copy()

    df_scaled_only = df.drop(['Close', 'High', 'Open', 'Low'], axis=1)
    df_scaled_only = rolling_zscore(df=df_scaled_only, cols=df_scaled_only.columns, window=30)

    df_model = df.copy()
    df_model[df_scaled_only.columns] = df_scaled_only[df_scaled_only.columns]
    
    df_model.columns = [x.lower() for x in df_model.columns]
    df_model.index.name = df_model.index.name.lower()

    df_model.drop(['dividends', 'stock splits'], axis=1, inplace=True)

    return df_model.dropna()


def rolling_zscore(df: pd.DataFrame, cols: list[str], window: int):
    roll = df[cols].rolling(window=window, min_periods=window)
    mu = roll.mean().shift(1)
    sigma = roll.std(ddof=0).replace(0, 1.0).shift(1)
    z = (df[cols] - mu) / sigma
    out = df.copy()
    out[cols] = z
    return out.dropna()

def zscore(df: pd.DataFrame, cols: list[str]):
    cols_to_scale = [c for c in cols if c not in {'Open', 'High', 'Low', 'Close', 'Volume'}]
    out = df.copy()

    mu = df[cols_to_scale].mean()
    sigma = df[cols_to_scale].std(ddof=0).replace(0, 1.0)

    out[cols_to_scale] = (df[cols_to_scale] - mu) / sigma
    #out.dropna().to_csv('sanity.csv', index=False)
    return out.dropna()

def minmax_scale(df: pd.DataFrame, cols: list[str]):
    # --- kolumny do skalowania (bez OHLCV) ---
    cols_to_scale = [c for c in cols if c not in {'Open', 'High', 'Low', 'Close', 'Volume'}]
    out = df.copy()

    # --- oblicz minimum i maksimum dla każdej kolumny ---
    min_vals = df[cols_to_scale].min()
    max_vals = df[cols_to_scale].max()

    # --- zabezpieczenie przed dzieleniem przez zero ---
    denom = (max_vals - min_vals).replace(0, 1.0)

    # --- skalowanie do [0, 1] ---
    out[cols_to_scale] = (df[cols_to_scale] - min_vals) / denom

    # --- zapis do sanity-check ---
    #out.dropna().to_csv('sanity.csv', index=False)
    return out.dropna()
