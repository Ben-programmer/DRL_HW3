"""HW3-2: Double DQN & Dueling DQN Page — player mode."""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.double_dqn import DoubleDQN
from models.dueling_dqn import DuelingDQN
from trainers.replay_trainer import train_with_target_network
from trainers.base_trainer import test_model, test_win_rate, Gridworld
from ui.components import render_gridworld_html, plot_losses, show_metrics


def render():
    st.header("HW3-2: Enhanced DQN Variants")

    st.markdown("""
    > **Task**: Implement **Double DQN** and **Dueling DQN** in `player` mode  
    > Compare how they improve upon the basic DQN
    """)

    # --- Concepts ---
    with st.expander("Concepts", expanded=False):
        st.markdown("""
        ### Double DQN
        - **Problem**: Standard DQN tends to **overestimate Q-values**.
        - **Solution**: Decouple action selection from Q-value evaluation.
          - **Online network** selects the best action `a* = argmax Q_online(s', a)`
          - **Target network** evaluates its Q-value `Q_target(s', a*)`

        ### Dueling DQN
        - **Problem**: Not all states require estimating the value of every action.
        - **Solution**: Decompose the Q-value into **V(s)** (state value) + **A(s,a)** (action advantage).
          - `Q(s,a) = V(s) + A(s,a) - mean(A(s,:))`
          - Helps the network learn which states are valuable more efficiently.
        """)

    # --- Model Selection ---
    model_type = st.radio(
        "Select Model",
        ["Double DQN", "Dueling DQN"],
        horizontal=True,
        key="hw32_model_type",
    )

    # --- Control Panel ---
    with st.expander("Training Parameters", expanded=True):
        epochs = st.slider("Epochs", 500, 10000, 5000, 500, key="hw32_epochs")
        lr = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3, key="hw32_lr")
        gamma = st.slider("Gamma", 0.5, 0.99, 0.9, 0.01, key="hw32_gamma")
        sync_freq = st.slider("Target Sync Freq", 100, 2000, 500, 100, key="hw32_sync")
        mem_size = st.slider("Replay Memory Size", 200, 5000, 1000, 100, key="hw32_mem")
        batch_size = st.slider("Batch Size", 32, 500, 200, 32, key="hw32_batch")

    # --- Buttons ---
    train_btn = st.button("Start Training", key="hw32_train", use_container_width=True)
    test_btn = st.button("Test Model (100 games)", key="hw32_test", use_container_width=True)

    key_prefix = "hw32_double" if model_type == "Double DQN" else "hw32_dueling"

    # --- Training ---
    if train_btn:
        if model_type == "Double DQN":
            model = DoubleDQN()
            use_double = True
        else:
            model = DuelingDQN()
            use_double = True  # Dueling also uses double strategy here

        st.session_state[f'{key_prefix}_model'] = model

        progress = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        all_losses = []

        def on_epoch(epoch, loss, epsilon):
            all_losses.append(loss)
            pct = (epoch + 1) / epochs
            progress.progress(pct)
            if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == epochs - 1:
                status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Epsilon: {epsilon:.3f}")
                chart_placeholder.plotly_chart(
                    plot_losses(all_losses, f"{model_type} — Loss Curve"),
                    use_container_width=True,
                )

        losses = train_with_target_network(
            model, epochs=epochs, mode='player', gamma=gamma,
            learning_rate=lr, sync_freq=sync_freq,
            mem_size=mem_size, batch_size=batch_size,
            use_double=use_double, callback=on_epoch,
        )
        st.session_state[f'{key_prefix}_losses'] = losses
        st.success(f"Training complete! Final loss: {losses[-1]:.6f}")

    # --- Testing ---
    if test_btn:
        model = st.session_state.get(f'{key_prefix}_model')
        if model is None:
            st.warning("Please train a model first.")
            return

        with st.spinner("Testing..."):
            win_rate, avg_moves, wins = test_win_rate(model, mode='player', num_games=100)
        show_metrics(win_rate, avg_moves, wins, 100)

        st.subheader("Sample Game")
        win, moves, path = test_model(model, mode='player')
        game = Gridworld(size=4, mode='player')
        st.markdown(render_gridworld_html(game.display()), unsafe_allow_html=True)
        st.markdown(f"**Result**: {'Win' if win else 'Lose'} in {moves} moves")
        st.markdown(f"**Path**: {' -> '.join(path)}")

    # Show existing losses
    if f'{key_prefix}_losses' in st.session_state and not train_btn:
        st.plotly_chart(
            plot_losses(st.session_state[f'{key_prefix}_losses'], f"{model_type} — Loss Curve"),
            use_container_width=True,
        )

