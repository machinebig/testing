import streamlit as st
from utils.api_client import APIClient

def display_logo():
    st.image("static/images/logo.png", width=200)

def dashboard_page():
    if "token" not in st.session_state:
        st.warning("Please log in")
        st.experimental_rerun()
    display_logo()
    st.title("Dashboard")
    client = APIClient()
    projects = client.get_projects(st.session_state["token"])
    st.write("Your Projects")
    st.dataframe(projects)
