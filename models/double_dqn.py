"""HW3-2: Double DQN Model for Gridworld."""
import torch
import torch.nn as nn


class DoubleDQN(nn.Module):
    """Double DQN — 結構同 NaiveDQN，但搭配 target network 使用。

    核心差異在訓練邏輯：
    - Online network 選擇動作 (argmax)
    - Target network 評估該動作的 Q 值
    避免 Q 值過度高估 (overestimation)。
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
