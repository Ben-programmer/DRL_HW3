"""共用 UI 元件：棋盤可視化、Loss 圖表、KPI 卡片。"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np


# --- Colors (Light Theme, Solid) ---
COLORS = {
    'P': '#3B82F6',  # Player - Blue
    '+': '#10B981',  # Goal   - Green
    '-': '#EF4444',  # Pit    - Red
    'W': '#6B7280',  # Wall   - Gray
    ' ': '#F8FAFC',  # Empty  - Very Light Gray
}

LABELS = {
    'P': 'Player',
    '+': 'Goal',
    '-': 'Pit',
    'W': 'Wall',
    ' ': '',
}

EMOJI = {
    'P': 'P',
    '+': 'G',
    '-': 'X',
    'W': 'W',
    ' ': '',
}


def render_gridworld_html(board_array):
    """Render Gridworld board as an HTML table."""
    size = board_array.shape[0]
    html = '<div style="display: flex; justify-content: center; padding: 20px;"><table style="border-collapse: collapse;">'
    for r in range(size):
        html += '<tr>'
        for c in range(size):
            cell = str(board_array[r, c]).strip()
            bg = COLORS.get(cell, '#FFFFFF')
            text = EMOJI.get(cell, '')
            font_color = '#FFFFFF' if cell != ' ' else '#000000'
            
            html += (
                f'<td style="width:70px;height:70px;text-align:center;'
                f'vertical-align:middle;border: 1px solid #E2E8F0;'
                f'background:{bg};font-size:24px;font-weight:bold;color:{font_color};">'
                f'{text}</td>'
            )
        html += '</tr>'
    html += '</table></div>'
    return html


def plot_losses(losses, title="Training Loss", height=380):
    """Plotly chart for losses (Light Theme)."""
    if not losses:
        return go.Figure()

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=losses,
        mode='lines',
        line=dict(color='#3B82F6', width=1.5),
        name='Loss',
    ))

    if len(losses) >= 50:
        window = min(50, len(losses) // 5)
        ma = np.convolve(losses, np.ones(window) / window, mode='valid')
        fig.add_trace(go.Scatter(
            x=list(range(window - 1, len(losses))),
            y=ma,
            mode='lines',
            line=dict(color='#F59E0B', width=2),
            name=f'MA ({window})',
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1E293B')),
        xaxis_title='Epoch',
        yaxis_title='Loss',
        height=height,
        template='plotly_white',
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(
            x=0.98, y=0.98, 
            xanchor='right', yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#E2E8F0',
            borderwidth=1,
        ),
        hovermode='x unified'
    )
    
    return fig


def show_metrics(win_rate, avg_moves, wins, total):
    """Display KPI metrics."""
    st.markdown(f"**Win Rate**: {win_rate:.1f}% | **Avg Moves**: {avg_moves:.1f} | **Wins / Total**: {wins} / {total}")


def inject_custom_css():
    """Inject custom light-theme CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    *, html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background-color: #FFFFFF;
        color: #0F172A;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: 1px solid #2563EB !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #2563EB !important;
        border-color: #1D4ED8 !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        margin-bottom: 1rem;
    }
    
    /* Code blocks */
    code {
        color: #D97706 !important;
        background: #FEF3C7 !important;
        padding: 2px 4px !important;
        border-radius: 4px !important;
    }
    
    /* Hide Top Bar */
    [data-testid="stHeader"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
