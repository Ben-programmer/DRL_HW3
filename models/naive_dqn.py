"""HW3-1: Naive DQN Model for Gridworld."""
import torch
import torch.nn as nn


class NaiveDQN(nn.Module):
    """基本的 DQN 網路 — 3 層全連接。

    結構: 64 → 150 → 100 → 4
    對應 Gridworld 4×4×4 狀態空間 → 4 個動作
    """

    def __init__(self, input_size=64, hidden1=150, hidden2=100, output_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size),
        )

    def forward(self, x):
        return self.net(x)
