import streamlit as st
from utils.api_client import APIClient

def project_view_page():
    if "token" not in st.session_state:
        st.warning("Please log in")
        st.experimental_rerun()
    st.title("Project View")
    project_id = st.selectbox("Select Project", [1, 2, 3])  # Placeholder
    client = APIClient()
    project = client.get_project(project_id, st.session_state["token"])
    st.write(f"Project: {project['name']}")
    st.write(project['description'])
