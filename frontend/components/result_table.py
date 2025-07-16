import streamlit as st
import pandas as pd

def display_result_table(df):
    st.dataframe(df)
