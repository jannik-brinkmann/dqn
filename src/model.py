import numpy as np
import torch
import torch.nn as nn


class DQNModel(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)
