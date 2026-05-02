"""HW3-1: Naive DQN Page — static mode."""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.naive_dqn import NaiveDQN
from trainers.naive_trainer import train_naive_dqn
from trainers.base_trainer import test_model, test_win_rate, Gridworld
from ui.components import render_gridworld_html, plot_losses, show_metrics


def render():
    st.header("HW3-1: Naive DQN")

    st.markdown("""
    > **Task**: Implement basic DQN in `static` mode  
    > Optional: **Experience Replay Buffer**  
    > Observe Training Loss and Win Rate
    """)

    # --- Control Panel ---
    with st.expander("Training Parameters", expanded=True):
        epochs = st.slider("Epochs", 100, 3000, 1000, 100, key="hw31_epochs")
        lr = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3, key="hw31_lr")
        gamma = st.slider("Gamma", 0.5, 0.99, 0.9, 0.01, key="hw31_gamma")
        eps_start = st.slider("Epsilon Start", 0.5, 1.0, 1.0, 0.05, key="hw31_eps_s")
        eps_end = st.slider("Epsilon End", 0.01, 0.3, 0.1, 0.01, key="hw31_eps_e")
        use_replay = st.checkbox("Use Experience Replay", value=False, key="hw31_replay")

        if use_replay:
            mem_size = st.slider("Replay Memory Size", 200, 5000, 1000, 100, key="hw31_mem")
            batch_size = st.slider("Batch Size", 32, 500, 200, 32, key="hw31_batch")
        else:
            mem_size = 1000
            batch_size = 200

    # --- Buttons ---
    train_btn = st.button("Start Training", key="hw31_train", use_container_width=True)
    test_btn = st.button("Test Model (100 games)", key="hw31_test", use_container_width=True)

    # --- Training ---
    if train_btn:
        model = NaiveDQN()
        st.session_state['hw31_model'] = model

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
                    plot_losses(all_losses, "HW3-1 Naive DQN — Loss Curve"),
                    use_container_width=True,
                )

        losses = train_naive_dqn(
            model, epochs=epochs, mode='static', gamma=gamma,
            learning_rate=lr, epsilon_start=eps_start, epsilon_end=eps_end,
            use_replay=use_replay, mem_size=mem_size, batch_size=batch_size,
            callback=on_epoch,
        )
        st.session_state['hw31_losses'] = losses
        st.success(f"Training complete! Final loss: {losses[-1]:.6f}")

    # --- Testing ---
    if test_btn:
        model = st.session_state.get('hw31_model')
        if model is None:
            st.warning("Please train a model first.")
            return

        with st.spinner("Testing..."):
            win_rate, avg_moves, wins = test_win_rate(model, mode='static', num_games=100)
        show_metrics(win_rate, avg_moves, wins, 100)

        # Show a sample game
        st.subheader("Sample Game")
        win, moves, path = test_model(model, mode='static')
        game = Gridworld(size=4, mode='static')
        st.markdown(render_gridworld_html(game.display()), unsafe_allow_html=True)
        st.markdown(f"**Result**: {'Win' if win else 'Lose'} in {moves} moves")
        st.markdown(f"**Path**: {' -> '.join(path)}")

    # Show existing losses
    if 'hw31_losses' in st.session_state and not train_btn:
        st.plotly_chart(
            plot_losses(st.session_state['hw31_losses'], "HW3-1 Naive DQN — Loss Curve"),
            use_container_width=True,
        )

