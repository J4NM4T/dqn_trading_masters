import torch
import pandas as pd
from src.mtrain import compute_metrics
from typing import List
import numpy as np

def evaluate_greedy(env, qnet, seed=123, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    was_training = qnet.training
    qnet.eval()
    obs, _ = env.reset(seed=seed)
    done = truncated = False
    reward_curve = []
    ep_return = 0.0

    equity_proxy = 1.0  # start NAV proxy at 1.0; reward is log-return (net of costs)

    ep_log: List[tuple[str, int, float, float, float]] = []  # (date_t, action, price_t, position_after, equity_proxy)


    while not (done or truncated):
        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            q = qnet(x)
            action = int(torch.argmax(q, dim=1).item())

        # log po kroku: stan "na koniec t" (env.step zwiększa step_idx)
        date_t   = str(env.df.index[env.step_idx])
        close_t  = float(env.df['Close'].iat[env.step_idx])
        
        obs, reward, done, truncated, info = env.step(action)

        position = float(env.position)

        ep_return += float(reward)
        reward_curve.append(float(reward))
        # compound with log-returns: NAV_t = NAV_{t-1} * exp(r_t)
        equity_proxy *= float(np.exp(reward))
        equity_proxy = float(np.clip(equity_proxy, 1e-12, 1e12))
        # dopiero teraz dopisujemy log (date, action, price, position, equity_proxy)

        ep_log.append((date_t, int(action), close_t, position, float(equity_proxy)))

    # --- tutaj wykorzystujemy Twoją istniejącą funkcję ---
    reward_series = pd.Series(reward_curve, dtype=float)
    metrics = compute_metrics(reward_series)

    # logi epizodu
    full_log = [ep_log[i] + (reward_curve[i],) for i in range(len(ep_log))]

    print( 
            f"| MaxDD: {metrics['MaxDD']*100:6.2f}% ",
            f"| Sharpe: {metrics['Sharpe']:5.2f} ",
            f"| Sortino: {metrics['Sortino']:5.2f} ",
            f"| Hit: {metrics['HitRatio']*100:5.1f}%"
    )

    list_dqns = pd.DataFrame([[metrics['MaxDD'], metrics['Sharpe'], metrics['Sortino'], metrics['HitRatio']]], columns=['MaxDD', 'Sharpe', 'Sortino', 'Hit'])

    if was_training:
        qnet.train()

    return full_log, list_dqns