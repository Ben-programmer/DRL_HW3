"""HW3-3: PyTorch Lightning DQN Model for Gridworld."""
import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl


class LightningDQN(pl.LightningModule):
    """用 PyTorch Lightning 封裝的 DQN。

    加入訓練技巧：
    - Gradient Clipping (在 Trainer 層級設定)
    - Learning Rate Scheduling (ReduceLROnPlateau 或 StepLR)
    - Huber Loss (SmoothL1Loss) 取代 MSE，更穩定
    """

    def __init__(
        self,
        input_size=64,
        hidden1=150,
        hidden2=100,
        output_size=4,
        lr=1e-3,
        gamma=0.9,
        use_huber_loss=True,
        lr_schedule_step=1000,
        lr_schedule_gamma=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.gamma = gamma

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size),
        )

        if use_huber_loss:
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_schedule_step,
            gamma=self.hparams.lr_schedule_gamma,
        )
        return [optimizer], [scheduler]
