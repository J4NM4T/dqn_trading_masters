import random
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import time

from src.env import TradingEnv
from src.model import QNetwork, DQNConfig, ReplayBuffer
from src.model import epsilon_by_step

def select_action(qnet: QNetwork, obs: np.ndarray, eps: float, n_actions: int, device: torch.device) -> int:
    if random.random() < eps:
        return random.randrange(n_actions)
    with torch.inference_mode():
        x = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        q = qnet(x)
        return int(torch.argmax(q, dim=1).item())

def compute_metrics(reward_curve: pd.Series):
    if reward_curve.empty:
        return {}
    
    r = pd.Series(reward_curve, dtype=float)
    equity = (1.0 + r).cumprod()

    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = drawdown.min()

    daily_returns = r 
    mean_ret = float(np.mean(daily_returns))
    std_ret  = float(np.std(daily_returns))
    sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252)
    if sharpe <= 1e-8:
        sharpe = 0
    if sharpe >= 200:
        sharpe = 200

    # Sortino (tylko ujemne odchylenia)
    neg = daily_returns[daily_returns < 0]
    if len(neg) == 0:
        sortino = float('inf')
    else:
        downside = float(np.std(neg))
        sortino = (mean_ret / (downside + 1e-9)) * np.sqrt(252)
        if sortino <= 1e-8:
            sortino = 0
        if sortino >= 200:
            sortino = 200

    # Hit ratio
    hit_ratio = float((daily_returns > 0).mean())

    return {
        "MaxDD": float(max_dd),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "HitRatio": float(hit_ratio)
    }

def train(env: TradingEnv, qcfg: DQNConfig, episodes: int, seed: int = 42,
          device: Optional[torch.device] = 'cpu'):
    # --- seedy ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path: str = f'dqns_{env.ticker}/dqn'

    # inicjalizacja sieci
    obs, _ = env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    qnet = QNetwork(obs_dim, n_actions).to(device)
    target = QNetwork(obs_dim, n_actions).to(device)

    target.load_state_dict(qnet.state_dict())

    opt = Adam(qnet.parameters(), lr=qcfg.lr)
    buf = ReplayBuffer(qcfg.buffer_size, obs_dim, seed)

    global_step = 0
    episode_returns: List[float] = []              # suma per-step rewards (proste stopy), diagnostycznie
    log_hist: List[tuple[float, float, float, float, float]] = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_return = 0.0                            # suma r_t (proste stopy)
        reward_curve: List[float] = []

        ep_log: List[tuple[str, int, float, int, float]] = []  # (date_t, action, price_t, position_after, equity_proxy)

        equity_proxy = 1.0
        while not (done or truncated):
            eps = epsilon_by_step(global_step, qcfg)
            action = select_action(qnet, obs, eps, n_actions, device)

            next_obs, reward, done, truncated, info = env.step(action)

            position = float(env.position)

            terminal = done or truncated

            buf.push(obs, action, reward, next_obs, terminal)

            ep_return += float(reward)
            reward_curve.append(float(reward))

            ep_log.append((info.get('date', None), int(action), float(info.get('close_price', np.nan)), position))

            obs = next_obs
            global_step += 1

            # uczenie po zapełnieniu bufora
            if len(buf) >= qcfg.min_buffer:
                s, a, r, ns, d = buf.sample(qcfg.batch_size)

                s  = torch.as_tensor(s, dtype=torch.float32, device=device)
                a  = torch.as_tensor(a, dtype=torch.int64, device=device)
                r  = torch.as_tensor(r, dtype=torch.float32, device=device)
                ns = torch.as_tensor(ns, dtype=torch.float32, device=device)
                d  = torch.as_tensor(d, dtype=torch.float32, device=device)

                q_vals = qnet(s)
                q_sa   = q_vals.gather(1, a.view(-1, 1)).squeeze(1)

                with torch.no_grad():

                    next_actions = qnet(ns).argmax(1, keepdim=True)
                    target_next_q = target(ns).gather(1, next_actions).squeeze(1)
                    target_q = r + qcfg.gamma * (1.0 - d) * target_next_q

                loss = F.smooth_l1_loss(q_sa, target_q)

                opt.zero_grad(set_to_none=True)
                loss.backward()

                if (global_step % 4) == 0:
                    nn.utils.clip_grad_norm_(qnet.parameters(), qcfg.grad_clip_norm)
                
                opt.step()

                if global_step % qcfg.target_update_interval == 0:
                    target.load_state_dict(qnet.state_dict())

        # logi epizodu
        full_log = [ep_log[i] + (reward_curve[i],) for i in range(len(ep_log))]
        
        episode_returns.append(ep_return)

        # metryki: reward_curve to r_t jako proste stopy, więc equity = (1+r).cumprod()
        rew = pd.Series(reward_curve, dtype=float)
        # --- Oblicz metryki ---
        metrics = compute_metrics(rew)

        # Zapisz do historii
        log_hist.append((
            ep_return,
            metrics['MaxDD'] * 100,
            metrics['Sharpe'],
            metrics['Sortino'],
            metrics['HitRatio']
        ))

        if ep % 50 == 0:
            print(
                f"Episode {ep:03d} | Sum(r_t): {ep_return:8.4f} "
                f"| MaxDD: {metrics['MaxDD']*100:6.2f}% "
                f"| Sharpe: {metrics['Sharpe']:5.2f} "
                f"| Sortino: {metrics['Sortino']:5.2f} "
                f"| Hit: {metrics['HitRatio']*100:5.1f}%"
                f"| Eps: {eps:5.2f}"
            )

        if ep % 100 == 0:
            torch.save(qnet.state_dict(), f'{save_path}_{ep}_{env.ticker}.pth')
            print(f"Zapisano wagi do: {save_path}_{ep}_{env.ticker}.pth")

    return qnet, episode_returns, full_log, log_hist
