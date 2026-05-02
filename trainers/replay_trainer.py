"""HW3-2: Double DQN / Dueling DQN 訓練器 (含經驗回放 + 目標網路)。"""
import copy
import random
import numpy as np
import torch
from collections import deque
from .base_trainer import get_state, select_action, ACTION_SET, Gridworld


def train_with_target_network(
    model,
    epochs=5000,
    mode='player',
    gamma=0.9,
    learning_rate=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.1,
    mem_size=1000,
    batch_size=200,
    max_moves=50,
    sync_freq=500,
    use_double=False,
    callback=None,
):
    """帶目標網路的訓練器，支援 Double DQN。

    Args:
        use_double: True=Double DQN, False=普通目標網路 DQN
        callback: fn(epoch, loss, epsilon) — UI 回呼
    Returns:
        losses: list of float
    """
    target_model = copy.deepcopy(model)
    target_model.load_state_dict(model.state_dict())

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
                # 隨機取樣 mini-batch
                minibatch = random.sample(list(replay), min(batch_size, len(replay)))
                states = torch.cat([s for s, a, r, s2, d in minibatch])
                actions = torch.LongTensor([a for s, a, r, s2, d in minibatch])
                rewards = torch.Tensor([r for s, a, r, s2, d in minibatch])
                next_states = torch.cat([s2 for s, a, r, s2, d in minibatch])
                dones = torch.Tensor([float(d) for s, a, r, s2, d in minibatch])

                with torch.no_grad():
                    if use_double:
                        # Double DQN: online 選動作，target 評估 Q
                        online_next_q = model(next_states)
                        best_actions = torch.argmax(online_next_q, dim=1)
                        target_next_q = target_model(next_states)
                        max_next_q = target_next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                    else:
                        target_next_q = target_model(next_states)
                        max_next_q = torch.max(target_next_q, dim=1).values

                Y = rewards + gamma * max_next_q * (1 - dones)

                current_qvals = model(states)
                X = current_qvals.gather(1, actions.unsqueeze(1)).squeeze(1)

                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()

            # 定期同步目標網路
            if j % sync_freq == 0:
                target_model.load_state_dict(model.state_dict())

            state1 = state2
            if done:
                status = 0

        losses.append(epoch_loss)
        if epsilon > epsilon_end:
            epsilon -= (epsilon_start - epsilon_end) / epochs

        if callback:
            callback(epoch, epoch_loss, epsilon)

    return losses
