import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradingEnvConfig:
    window: int = 60
    trading_cost_bps: float = 2.0
    reward_scale: float = 1.0
    max_episode_steps: Optional[int] = 126
    positions: tuple[float, ...] = (-1.0, 0.0, 1.0)

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, ticker: str, df: pd.DataFrame, feature_cols: list[str],
                 config: TradingEnvConfig = TradingEnvConfig(), seed: Optional[int] = None):
        super().__init__()

        self.ticker = ticker
        self.df = df
        self.feature_cols = feature_cols
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        self.window = self.cfg.window
        # Prekomputacje Numpy (bez Pandas w pętli)
        base_features = self.df[self.feature_cols].drop('Close', axis=1, errors='ignore')
        self.features_np = base_features.to_numpy(dtype=np.float32)
        self.feature_dim = self.features_np.shape[1]
        self.close_np = self.df['Close'].to_numpy(dtype=np.float32)
        # daty jako stringi do logów/info
        self.date_arr = self.df.index.astype(str)

        self.extra_dim = 1
        obs_dim = self.window * self.feature_dim + self.extra_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.positions = np.array(sorted(set(float(x) for x in self.cfg.positions)), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.positions))

        self._reset_state()

    def _reset_state(self):
        if self.cfg.max_episode_steps is not None:
            max_start = len(self.df) - (self.cfg.max_episode_steps + 1)
        else:
            max_start = len(self.df) - 2

        # Losowy punkt startowy epizodu
        self.step_idx = int(self.rng.integers(
            low=self.window,
            high=max_start
        ))

        # Licznik kroków epizodu
        self.episode_steps = 0

        self.position = 0.0
        self.last_price = float(self.df['Close'].iat[self.step_idx - 1])
        self.terminated = False
        self.truncated = False

    def _mk_obs(self) -> np.ndarray:
        start, end = self.step_idx - self.window, self.step_idx
        obs = self.features_np[start:end].reshape(-1)
        exposure_frac = np.float32(self.position)
        obs = np.concatenate([obs, [exposure_frac]], axis=0)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        obs = self._mk_obs()
        date = str(self.date_arr[self.step_idx - 1])
        info = {"date": date, "position": float(self.position)}

        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)

        # CLOSE_t = bieżąca cena
        close_t = float(self.close_np[self.step_idx])

        # CLOSE_{t+1} musi istnieć – jeśli nie istnieje, to terminal
        if self.step_idx + 1 >= len(self.df):
            # Naturalne zakończenie danych
            obs = self._mk_obs()
            info = {"date": None, "position": float(self.position)}
            return obs, 0.0, True, False, info

        close_tp1 = float(self.close_np[self.step_idx + 1])

        f_t = float(self.positions[action])
        f_prev = float(self.position)

        bar_ret = np.log(close_tp1 / close_t)
        bar_pnl = f_t * bar_ret

        turnover = abs(f_t - f_prev)
        cost_rate = (self.cfg.trading_cost_bps / 1e4)
        trade_cost = turnover * cost_rate

        net_ret = bar_pnl - trade_cost
        reward = float(net_ret * self.cfg.reward_scale)

        # aktualizacja stanu
        self.position = f_t
        self.step_idx += 1
        self.episode_steps += 1

        # Prawdziwy terminal (koniec danych)
        terminated = (self.step_idx >= len(self.df) - 1)

        # Sztuczne ucięcie epizodu (limit kroków)
        truncated = (
            self.cfg.max_episode_steps is not None
            and self.episode_steps >= self.cfg.max_episode_steps
        )

        obs = self._mk_obs()
        date = str(self.date_arr[self.step_idx]) if self.step_idx < len(self.df) else None

        info = {
            "date": date,
            "position": float(self.position),
            "bar_ret": float(bar_ret),
            "turnover": float(turnover),
            "cost_frac": float(trade_cost),
            "trade_cost": float(trade_cost),
            "net_ret": float(net_ret),
            "close_price": float(close_t),
        }

        return obs, reward, terminated, truncated, info