import streamlit as st
from streamlit_lottie import st_lottie
import json
import requests
import front_designing

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

st.set_page_config(page_title="Vihaan-Feedback", page_icon=":memo:", layout="wide")
front_designing.side_nav_bar()

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 80px !important;  # Set the width to your desired value
    }
</style>
""", unsafe_allow_html=True)

with st.container():
    left_column,right_column=st.columns(2)
    with left_column:
        st.header(":mailbox: Give us your feedback!")
        st.subheader("“Your feedback is invaluable to us. It aids in enhancing both our website and user experience.”")
    with right_column:
        lottie_feedback=load_lottiefile("./json_lottie/feedback.json")
        st_lottie(lottie_feedback, height=200, key="feedback")
feedback_form="""
<form action="https://formsubmit.co/deepika.sharma90559@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your Name" required>
     <input type="email" name="email" placeholder="Your Email" required>
     <textarea name="message" placeholder="Your Feedback"></textarea>
     <button type="submit">Send</button>
</form>
"""
st.markdown(feedback_form, unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/feedbackstyle.css")