import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from collections import deque
import random
import numpy as np
from typing import Optional, Tuple

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 200_000
    min_buffer: int = 20_000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 150_000
    grad_clip_norm: float = 1.0
    target_update_interval: int = 10_000

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_sizes: Tuple[int, int] = (64, 64)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1 = nn.Linear(input_dim, h1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(h2, n_actions)
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: Optional[int] = None):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.states = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_states = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity,), dtype=np.int64)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.dones = np.empty((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, ns, d):
        self.states[self.pos] = s
        self.actions[self.pos] = a
        self.rewards[self.pos] = r
        self.next_states[self.pos] = ns
        self.dones[self.pos] = d
        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        s = self.states[idx]
        a = self.actions[idx]
        r = self.rewards[idx]
        ns = self.next_states[idx]
        d = self.dones[idx]
        return s, a, r, ns, d

    def __len__(self):
        return self.size


def epsilon_by_step(step: int, cfg: DQNConfig) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    frac = step / cfg.eps_decay_steps
    eps = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac

    return float(eps)