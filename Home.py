import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from streamlit_lottie import st_lottie
import json
import requests
import front_designing
import pickle
import streamlit_authenticator as stauth
from pathlib import Path

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
# st.markdown("""
# <style>
#     section[data-testid="stSidebar"] {
#         width: 80px !important;  # Set the width to your desired value
#     }
# </style>
# """, unsafe_allow_html=True)

# margins_css = """
# <style>
#     .main > div {
#         padding-left: 0rem;
#         padding-right: 0rem;
#     }
# </style>
# """
# st.markdown(margins_css, unsafe_allow_html=True)
st.set_page_config(page_title="Vihaan-Home", page_icon=":speech_balloon:", layout="wide")
front_designing.side_nav_bar()

with st.container():
    # st.write("---")
    st.subheader(f"Hi, I am Vihaan :wave:")
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Welcome to our AI-Powered Fashion Companion!")
        st.write("Are you tired of sifting through countless fashion options online, feeling overwhelmed by the sheer variety? Look no further! Our cutting-edge fashion recommendation system is here to revolutionize your shopping experience.")
    with right_column:
        lottie_ai=load_lottiefile("./json_lottie/ai.json")
        st_lottie(lottie_ai, height=300, key="ai")

with st.container():
    left_column,mid_column,right_column=st.columns([0.5,3,0.5])
    with mid_column:
        st.video("vihaan-video.mp4",start_time=0)
with st.container():
    st.write("---")
    st.title("What Our Website Has to Offer?")
    left_column, right_column = st.columns(2)
    with left_column:
        lottie_choosingmen=load_lottiefile("./json_lottie/choosingmen.json")
        st_lottie(lottie_choosingmen, height=250, key="choosingmen")
    with right_column:
        st.title("Discover Your Perfect Look ")
        st.write("Are you tired of endlessly scrolling through fashion websites, hoping to find that perfect dress or accessory? Vihaan is here to simplify your style journey!")
        
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Find Similar Styles")
        st.write("Upload an image of a dress or accessory you love, and let Vihaan work its magic. Our advanced recommendation system analyzes your preferences and suggests similar or related items. Say goodbye to endless searches—Vihaan saves you time and helps you discover new favorites effortlessly.")
    with right_column:
        lottie_uploading2=load_lottiefile("./json_lottie/uploading2.json")
        st_lottie(lottie_uploading2, height=250, key="uploading2")

with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        lottie_wardrobe=load_lottiefile("./json_lottie/wardrobe.json")
        st_lottie(lottie_wardrobe, height=250, key="wardrobe")
    with right_column:
        st.title("Revamp Your Wardrobe")
        st.write("Got a closet full of clothes but struggle to put together fresh outfits? Vihaan has you covered! Upload pictures of your wardrobe pieces, and our intelligent engine will create stylish combinations for any occasion. Plus, Vihaan keeps track of your recent outfits, ensuring you never repeat the same look twice.")
    
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Why Choose Vihaan?")
    with right_column:
        lottie_select=load_lottiefile("./json_lottie/select.json")
        st_lottie(lottie_select, height=50, key="select")

with st.container():
    left_column, right_column = st.columns([1,2])
    with left_column:
        lottie_recommending=load_lottiefile("./json_lottie/recommending.json")
        st_lottie(lottie_recommending, height=200, key="recommending")
    with right_column:
        st.subheader("Personalized Recommendations")
        st.write(" Vihaan understands your unique style and tailors suggestions just for you.")
        

with st.container():
    left_column, right_column = st.columns([1,2])
    with left_column:
        lottie_save_time=load_lottiefile("./json_lottie/save_time.json")
        st_lottie(lottie_save_time, height=200, key="save_time")

    with right_column:
        st.subheader("Time-Saving")
        st.write("No more aimless browsing—get curated options instantly.")
        
with st.container():
    left_column, right_column = st.columns([1,2])
    with left_column:
        lottie_remix=load_lottiefile("./json_lottie/remix.json")
        st_lottie(lottie_remix, height=200, key="remix")
    with right_column:
        st.subheader("Wardrobe Remix")
        st.write("Turn old favorites into exciting new ensembles.")
        

with st.container():
    left_column, right_column = st.columns([1,2])
    with left_column:
        lottie_no=load_lottiefile("./json_lottie/no.json")
        st_lottie(lottie_no, height=200, key="no")
    with right_column:
        st.subheader("No More Fashion Faux Pas")
        st.write("Vihaan ensures you’re always on-trend.")

