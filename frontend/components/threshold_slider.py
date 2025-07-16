import streamlit as st

def threshold_slider(label, min_value=0.0, max_value=1.0, default_value=0.7):
    return st.slider(label, min_value, max_value, default_value)
