"""HW3-1: Naive DQN 訓練器 (含可選經驗回放)。"""
import numpy as np
import torch
from collections import deque
from .base_trainer import get_state, select_action, ACTION_SET, Gridworld


def train_naive_dqn(
    model,
    epochs=1000,
    mode='static',
    gamma=0.9,
    learning_rate=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.1,
    use_replay=False,
    mem_size=1000,
    batch_size=200,
    max_moves=50,
    callback=None,
):
    """訓練 Naive DQN。

    Args:
        callback: fn(epoch, loss, epsilon) — 每 epoch 結束時回呼，用於 UI 更新
    Returns:
        losses: list of float
    """
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epsilon = epsilon_start
    losses = []
    replay = deque(maxlen=mem_size) if use_replay else None

    for epoch in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state1 = get_state(game)
        status = 1
        mov = 0
        epoch_loss = 0.0
        step_count = 0

        while status == 1:
            mov += 1
            action, qval = select_action(model, state1, epsilon)
            game.makeMove(ACTION_SET[action])
            state2 = get_state(game)
            reward = game.reward()

            if use_replay:
                # 經驗回放
                replay.append((state1, action, reward, state2, reward != -1))
                if len(replay) >= batch_size:
                    batch = list(replay)[-batch_size:]  # 取最近的 batch
                    # 拆分 batch
                    states = torch.cat([s for s, a, r, s2, d in batch])
                    actions = torch.LongTensor([a for s, a, r, s2, d in batch])
                    rewards = torch.Tensor([r for s, a, r, s2, d in batch])
                    next_states = torch.cat([s2 for s, a, r, s2, d in batch])
                    dones = torch.Tensor([float(d) for s, a, r, s2, d in batch])

                    with torch.no_grad():
                        next_qvals = model(next_states)
                    max_next_q = torch.max(next_qvals, dim=1).values
                    Y = rewards + gamma * max_next_q * (1 - dones)

                    current_qvals = model(states)
                    X = current_qvals.gather(1, actions.unsqueeze(1)).squeeze(1)

                    loss = loss_fn(X, Y.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss = loss.item()
            else:
                # 無經驗回放 (原始 Naive DQN)
                with torch.no_grad():
                    newQ = model(state2.reshape(1, 64))
                maxQ = torch.max(newQ)
                if reward == -1:
                    Y = reward + (gamma * maxQ)
                else:
                    Y = torch.Tensor([reward])
                Y = torch.Tensor([Y]).detach()
                X = qval.squeeze()[action]
                loss = loss_fn(X, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()

            state1 = state2
            if abs(reward) == 10 or mov > max_moves:
                status = 0

        losses.append(epoch_loss)
        if epsilon > epsilon_end:
            epsilon -= (epsilon_start - epsilon_end) / epochs

        if callback:
            callback(epoch, epoch_loss, epsilon)

    return losses
