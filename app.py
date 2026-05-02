"""HW3: DQN and its Variants — Streamlit Interactive Application."""
import streamlit as st
import sys
import os

# Ensure modules can be imported
sys.path.insert(0, os.path.dirname(__file__))

from ui.components import inject_custom_css

# --- Page Config ---
st.set_page_config(
    page_title="HW3 — DQN and its Variants",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_custom_css()

# --- Custom Centered Header ---
st.markdown(
    '<h1 style="text-align: center; font-size: 3.4rem; font-weight: bold; margin-bottom: 2rem;">DQN Homework3</h1>',
    unsafe_allow_html=True
)

# --- Navigation ---
page = st.selectbox(
    "Select Section",
    [
        "HW3-1: Naive DQN",
        "HW3-2: Enhanced DQN",
        "HW3-3: Lightning DQN",
    ],
    key="page_select",
)

st.markdown("""
<div style="background-color: #F8FAFC; padding: 1.2rem 1.5rem; border-radius: 8px; border: 1px solid #E2E8F0; margin: 1rem 0;">
    <h4 style="margin-top: 0; color: #1E293B; font-size: 1.1rem;">Gridworld Environment</h4>
    <ul style="margin-bottom: 0; color: #334155;">
        <li>4x4 grid</li>
        <li>Player | Goal | Pit | Wall</li>
        <li>Actions: Up / Down / Left / Right</li>
        <li>Reward: +10 (Goal), -10 (Pit), -1 (else)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --- Route to selected page ---
if page == "HW3-1: Naive DQN":
    from ui.hw3_1_page import render
    render()
elif page == "HW3-2: Enhanced DQN":
    from ui.hw3_2_page import render
    render()
elif page == "HW3-3: Lightning DQN":
    from ui.hw3_3_page import render
    render()

