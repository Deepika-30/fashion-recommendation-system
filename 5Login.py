import streamlit as st
import pickle
import streamlit_authenticator as stauth
from pathlib import Path

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 80px !important;  # Set the width to your desired value
    }
</style>
""", unsafe_allow_html=True)

#--user authentication
names= ["Deepika Sharma", "Ashish Singh"]
usernames= ["dsharma", "asingh"]

#load hashed passwords
file_path= Path(__file__).parent.parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords=pickle.load(file)

authenticator=stauth.Authenticate(names, usernames, hashed_passwords, "Vihaan_dashboard","abcdef" )

name, authentication_status,username = authenticator.login("Login", "main")

if authentication_status==False:
    st.error("Username/password is incorrect")

if authentication_status== None:
    st.warning("Please enter your username and password")

