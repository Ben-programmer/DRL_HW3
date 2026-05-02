"""HW3-3: PyTorch Lightning DQN 訓練器。"""
import copy
import random
import numpy as np
import torch
from collections import deque
from .base_trainer import get_state, select_action, ACTION_SET, Gridworld


def train_lightning_dqn(
    model,
    epochs=5000,
    mode='random',
    gamma=0.9,
    learning_rate=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.1,
    mem_size=1000,
    batch_size=200,
    max_moves=50,
    sync_freq=500,
    grad_clip_val=1.0,
    use_huber_loss=True,
    callback=None,
):
    """用 PyTorch Lightning 模型的手動訓練迴圈。

    訓練技巧：
    1. Gradient Clipping — 避免梯度爆炸
    2. LR Scheduling — 自動衰減學習率
    3. Huber Loss (SmoothL1) — 比 MSE 更穩定
    """
    target_model = copy.deepcopy(model)
    target_model.load_state_dict(model.state_dict())

    if use_huber_loss:
        loss_fn = torch.nn.SmoothL1Loss()
    else:
        loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    epsilon = epsilon_start
    losses = []
    replay = deque(maxlen=mem_size)
    j = 0

    for epoch in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state1 = get_state(game)
        status = 1
        mov = 0
        epoch_loss = 0.0

        while status == 1:
            j += 1
            mov += 1
            action, qval = select_action(model, state1, epsilon)
            game.makeMove(ACTION_SET[action])
            state2 = get_state(game)
            reward = game.reward()
            done = (reward != -1) or (mov > max_moves)

            replay.append((state1, action, reward, state2, done))

            if len(replay) >= batch_size:
                minibatch = random.sample(list(replay), min(batch_size, len(replay)))
                states = torch.cat([s for s, a, r, s2, d in minibatch])
                actions = torch.LongTensor([a for s, a, r, s2, d in minibatch])
                rewards = torch.Tensor([r for s, a, r, s2, d in minibatch])
                next_states = torch.cat([s2 for s, a, r, s2, d in minibatch])
                dones = torch.Tensor([float(d) for s, a, r, s2, d in minibatch])

                with torch.no_grad():
                    # Double DQN 策略
                    online_next_q = model(next_states)
                    best_actions = torch.argmax(online_next_q, dim=1)
                    target_next_q = target_model(next_states)
                    max_next_q = target_next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)

                Y = rewards + gamma * max_next_q * (1 - dones)
                current_qvals = model(states)
                X = current_qvals.gather(1, actions.unsqueeze(1)).squeeze(1)

                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()

                # Gradient Clipping
                if grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

                optimizer.step()
                epoch_loss = loss.item()

            if j % sync_freq == 0:
                target_model.load_state_dict(model.state_dict())

            state1 = state2
            if done:
                status = 0

        scheduler.step()
        losses.append(epoch_loss)
        if epsilon > epsilon_end:
            epsilon -= (epsilon_start - epsilon_end) / epochs

        if callback:
            current_lr = optimizer.param_groups[0]['lr']
            callback(epoch, epoch_loss, epsilon, current_lr)

    return losses
