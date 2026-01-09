from typing import List
import pandas as pd
import matplotlib.pyplot as plt

def plot_last_episode(
    ticker: str,
    df_raw: pd.DataFrame,
    ep_log: List[tuple[str, int, float, int, float]],
):
    # 1) Log do ramki i wyrównanie czasu
    try:
        cols = ["Date", "action", "close_t", "position", "equity", "reward_t"]
        log_df = pd.DataFrame(ep_log, columns=cols)
    except:
        cols = ["Date", "action", "close_t", "position", "reward_t"]
        log_df = pd.DataFrame(ep_log, columns=cols)
    log_df["Date"] = pd.to_datetime(log_df["Date"])  # bezpieczeństwo
    log_df = log_df.set_index("Date").sort_index()

    log_df.to_csv(f'log_data/log_{ticker}.csv')

    # 2) Podzbiór danych surowych pod zakres epizodu
    start_date, end_date = log_df.index.min(), log_df.index.max()
    df = df_raw.copy()
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.loc[start_date:end_date].copy()

    fig, ax1 = plt.subplots(figsize=(18, 10))

    # Plot the trades
    buy_trades = log_df[(log_df['position'] >= 1)]
    sell_trades = log_df[(log_df['position'] == -1)]

    # Plot buy and sell trades as scatter plots on the price chart
    ax1.scatter(sell_trades.index, sell_trades['close_t'], s=50, color='red', alpha=0.3, label='Sell Trades')
    ax1.scatter(buy_trades.index, buy_trades['close_t'], s=50, color='green', alpha=0.3, label='Buy Trades')

    # Plot the Bitcoin price data
    ax1.plot(log_df['close_t'], label=f'{ticker} Price', color='red', linewidth=0.5, linestyle="--")

    # Labels for the plot
    ax1.set_title(f'{ticker} Price and trades')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()