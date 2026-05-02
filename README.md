# Deep Reinforcement Learning — HW3: DQN and its Variants

An interactive web application built with Streamlit to visualize and train Deep Q-Networks (DQN) and its advanced variants on a custom Gridworld environment.

## 🌟 Overview

This project implements various DQN architectures to solve a 4x4 Gridworld problem. The application provides an interactive dashboard allowing users to tweak hyperparameters (Epochs, Learning Rate, Gamma, Batch Size, etc.), train models in real-time, and visualize the agent's learning process through interactive Loss curves and dynamic Gridworld pathfinding.

## ✨ Features

*   **Interactive Streamlit Dashboard**: A clean, single-column UI with a modern light theme for a seamless analytical experience.
*   **Custom Gridworld Environment**: A 4x4 grid featuring a Player (`P`), Goal (`G`), Pit (`X`), and Wall (`W`). The environment supports static, player-initialized, and fully random modes.
*   **HW3-1: Naive DQN (Static Mode)**:
    *   Basic DQN implementation.
    *   Optional integration of an Experience Replay Buffer to stabilize training.
*   **HW3-2: Enhanced DQN Variants (Player Mode)**:
    *   **Double DQN**: Mitigates the overestimation of Q-values by decoupling action selection (Online Network) and evaluation (Target Network).
    *   **Dueling DQN**: Decomposes Q-values into State Value `V(s)` and Action Advantage `A(s,a)` to help the network learn valuable states more efficiently.
*   **HW3-3: PyTorch Lightning DQN (Random Mode)**:
    *   Modularized and scalable DQN training using PyTorch Lightning.
    *   Integration of advanced training techniques: 
        *   **Huber Loss (Smooth L1 Loss)**: Robustness against outliers.
        *   **Gradient Clipping**: Prevents exploding gradients.
        *   **Learning Rate Scheduling**: Utilizes `StepLR` for dynamic learning rate decay.

## 📂 Project Structure

```text
.
├── app.py                   # Main Streamlit application entry point
├── HW3_Info.md              # Original assignment instructions
├── README.md                # Project documentation
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration (Light theme)
├── gridworld/               # Gridworld environment logic
│   ├── GridBoard.py         # Grid board representation
│   └── Gridworld.py         # Game logic and state management
├── models/                  # Neural network architectures
│   ├── naive_dqn.py
│   ├── double_dqn.py
│   ├── dueling_dqn.py
│   └── lightning_dqn.py
├── trainers/                # Training loops and evaluation logic
│   ├── base_trainer.py      # Shared testing and evaluation functions
│   ├── naive_trainer.py
│   ├── replay_trainer.py    # Target network synchronization logic
│   └── lightning_trainer.py # Lightning Trainer wrapper
└── ui/                      # Streamlit UI components and pages
    ├── components.py        # Shared UI (Grid renderer, Charts, CSS)
    ├── hw3_1_page.py
    ├── hw3_2_page.py
    └── hw3_3_page.py
```

## 🚀 Installation

1.  **Navigate to the project directory**:
    ```bash
    cd /path/to/your/project/HW3
    ```

2.  **Create and activate a virtual environment** (Recommended using Anaconda):
    ```bash
    conda create -n drl_env python=3.10
    conda activate drl_env
    ```

3.  **Install the required dependencies**:
    ```bash
    pip install torch torchvision torchaudio
    pip install pytorch-lightning
    pip install streamlit plotly numpy
    ```

## 🖥️ Usage

To launch the interactive dashboard, run the following command in your terminal from the root directory of the project:

```bash
streamlit run app.py
```

### Dashboard Navigation

1.  **Select Section**: Use the dropdown menu at the top to navigate between the three homework sections (HW3-1, HW3-2, and HW3-3).
2.  **Training Parameters**: Expand the "Training Parameters" panel to adjust hyperparameters such as Epochs, Learning Rate, Gamma, and Target Sync Frequency.
3.  **Start Training**: Click to begin the training process. A progress bar and a real-time Plotly chart will track the Loss and epsilon decay.
4.  **Test Model**: After training, click to test the agent over 100 games to see its Win Rate, Average Moves, and a step-by-step path visualization rendered directly on the Gridworld UI.

## 🛠️ Technologies Used

*   **Python 3**
*   **PyTorch**: Core deep learning framework.
*   **PyTorch Lightning**: High-level interface for structuring PyTorch training code.
*   **Streamlit**: Web framework for interactive and data-driven UIs.
*   **Plotly**: Interactive charting and data visualization.
*   **NumPy**: Array manipulation and calculations.
