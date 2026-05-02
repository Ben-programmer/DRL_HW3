"""HW3-2: Dueling DQN Model for Gridworld."""
import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """Dueling DQN — 分離 Value Stream 和 Advantage Stream。

    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    這使得網路可以更好地學習哪些狀態是好的/壞的，
    而不必同時學習每個動作在每個狀態下的效果。
    """

    def __init__(self, input_size=64, hidden1=150, hidden2=100, output_size=4):
        super().__init__()
        # 共享特徵層
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
        )
        # Value stream: V(s) — 狀態價值
        self.value_stream = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        # Advantage stream: A(s,a) — 動作優勢
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)           # (batch, 1)
        advantage = self.advantage_stream(features)   # (batch, 4)
        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values
