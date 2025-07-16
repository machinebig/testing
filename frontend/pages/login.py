import streamlit as st
from utils.api_client import APIClient

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        client = APIClient()
        response = client.login(username, password)
        if response.get("access_token"):
            st.session_state["token"] = response["access_token"]
            st.success("Logged in successfully!")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
