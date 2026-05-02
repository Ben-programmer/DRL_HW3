"""HW3-3: PyTorch Lightning DQN Page — random mode + training techniques."""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.lightning_dqn import LightningDQN
from trainers.lightning_trainer import train_lightning_dqn
from trainers.base_trainer import test_model, test_win_rate, Gridworld
from ui.components import render_gridworld_html, plot_losses, show_metrics


def render():
    st.header("HW3-3: PyTorch Lightning DQN")

    st.markdown("""
    > **Task**: Wrap DQN using **PyTorch Lightning** and train in `random` mode.  
    > Implement Training Techniques: **Gradient Clipping**, **LR Scheduling**, **Huber Loss**
    """)

    # --- Training Techniques ---
    with st.expander("Training Techniques", expanded=False):
        st.markdown("""
        ### Gradient Clipping
        Limits the maximum norm of gradients to prevent exploding gradients and stabilize training.

        ### Learning Rate Scheduling
        Uses `StepLR` to multiply the learning rate by a decay factor every 1000 steps, gradually lowering the learning rate as training progresses.

        ### Huber Loss (SmoothL1Loss)
        - Acts like MSE when the error is small (smooth).
        - Acts like MAE when the error is large (no explosion).
        - More robust to outliers compared to standard MSE.
        """)

    # --- Control Panel ---
    with st.expander("Training Parameters", expanded=True):
        epochs = st.slider("Epochs", 500, 10000, 5000, 500, key="hw33_epochs")
        lr = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3, key="hw33_lr")
        gamma = st.slider("Gamma", 0.5, 0.99, 0.9, 0.01, key="hw33_gamma")
        grad_clip = st.slider("Gradient Clip Value", 0.0, 5.0, 1.0, 0.5, key="hw33_clip")
        use_huber = st.checkbox("Use Huber Loss", value=True, key="hw33_huber")
        sync_freq = st.slider("Target Sync Freq", 100, 2000, 500, 100, key="hw33_sync")
        mem_size = st.slider("Replay Memory Size", 200, 5000, 1000, 100, key="hw33_mem")
        batch_size = st.slider("Batch Size", 32, 500, 200, 32, key="hw33_batch")

    # --- Buttons ---
    train_btn = st.button("Start Training", key="hw33_train", use_container_width=True)
    test_btn = st.button("Test Model (100 games)", key="hw33_test", use_container_width=True)

    # --- Training ---
    if train_btn:
        model = LightningDQN(lr=lr, gamma=gamma, use_huber_loss=use_huber)
        st.session_state['hw33_model'] = model

        progress = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        lr_placeholder = st.empty()
        all_losses = []
        lr_history = []

        def on_epoch(epoch, loss, epsilon, current_lr):
            all_losses.append(loss)
            lr_history.append(current_lr)
            pct = (epoch + 1) / epochs
            progress.progress(pct)
            if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == epochs - 1:
                status_text.text(
                    f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | "
                    f"Epsilon: {epsilon:.3f} | LR: {current_lr:.6f}"
                )
                chart_placeholder.plotly_chart(
                    plot_losses(all_losses, "Lightning DQN — Loss Curve"),
                    use_container_width=True,
                )

        losses = train_lightning_dqn(
            model, epochs=epochs, mode='random', gamma=gamma,
            learning_rate=lr, grad_clip_val=grad_clip,
            use_huber_loss=use_huber, sync_freq=sync_freq,
            mem_size=mem_size, batch_size=batch_size,
            callback=on_epoch,
        )
        st.session_state['hw33_losses'] = losses
        st.success(f"Training complete! Final loss: {losses[-1]:.6f}")

        # Show Training Summary
        st.info(f"""
        **Training Summary**
        - Loss function: {'Huber (SmoothL1)' if use_huber else 'MSE'}
        - Gradient clipping: {grad_clip if grad_clip > 0 else 'Disabled'}
        - LR scheduling: StepLR (step=1000, gamma=0.5)
        - Final LR: {lr_history[-1]:.6f} (started at {lr})
        """)

    # --- Testing ---
    if test_btn:
        model = st.session_state.get('hw33_model')
        if model is None:
            st.warning("Please train a model first.")
            return

        with st.spinner("Testing..."):
            win_rate, avg_moves, wins = test_win_rate(model, mode='random', num_games=100)
        show_metrics(win_rate, avg_moves, wins, 100)

        st.subheader("Sample Game")
        win, moves, path = test_model(model, mode='random')
        game = Gridworld(size=4, mode='random')
        st.markdown(render_gridworld_html(game.display()), unsafe_allow_html=True)
        st.markdown(f"**Result**: {'Win' if win else 'Lose'} in {moves} moves")
        st.markdown(f"**Path**: {' -> '.join(path)}")

    # Show existing losses
    if 'hw33_losses' in st.session_state and not train_btn:
        st.plotly_chart(
            plot_losses(st.session_state['hw33_losses'], "Lightning DQN — Loss Curve"),
            use_container_width=True,
        )

