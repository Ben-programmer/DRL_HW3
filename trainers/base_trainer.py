"""共用的訓練與測試邏輯。"""
import sys
import os
import numpy as np
import torch
import random
from collections import deque

# 將 gridworld 目錄加入搜尋路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gridworld'))
from Gridworld import Gridworld

ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
ACTION_NAMES = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}


def get_state(game, noise=0.01):
    """取得遊戲狀態張量。"""
    state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) * noise
    return torch.from_numpy(state).float()


def select_action(model, state, epsilon):
    """ε-greedy 策略選擇動作。"""
    qval = model(state)
    qval_np = qval.data.numpy()
    if random.random() < epsilon:
        action = np.random.randint(0, 4)
    else:
        action = np.argmax(qval_np)
    return action, qval


def test_model(model, mode='static', max_moves=15):
    """測試模型，回傳 (是否勝利, 移動步數, 移動路徑)。"""
    game = Gridworld(size=4, mode=mode)
    state = get_state(game)
    moves = []
    status = 1
    i = 0

    while status == 1:
        qval = model(state)
        qval_np = qval.data.numpy()
        action = np.argmax(qval_np)
        action_name = ACTION_SET[action]
        moves.append(ACTION_NAMES[action])

        game.makeMove(action_name)
        state = get_state(game)
        reward = game.reward()

        if reward != -1:
            status = 2 if reward > 0 else 0

        i += 1
        if i > max_moves:
            break

    win = (status == 2)
    return win, i, moves


def test_win_rate(model, mode='random', num_games=100):
    """測試勝率。"""
    wins = 0
    total_moves = 0
    for _ in range(num_games):
        win, moves, _ = test_model(model, mode=mode)
        if win:
            wins += 1
        total_moves += moves
    win_rate = wins / num_games * 100
    avg_moves = total_moves / num_games
    return win_rate, avg_moves, wins
