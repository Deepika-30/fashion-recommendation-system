import streamlit as st
def side_nav_bar():
    side_bar="""
    
    <style>
    .st-emotion-cache-vk3wp9 {
    max-width: 150px; /* Adjust the maximum width to your desired value */
    width: auto; /* Allow the width to adjust based on content */
    }
    
    </style>
    """
    st.markdown(side_bar,unsafe_allow_html=True)

def drop_box():
    box="""
    <style>
    .st-emotion-cache-1gulkj5 {
    height: 200px; /* Adjust the height to your desired value */
    width: 70%;
    background-color: Lavender;
    margin: 0 auto; 
    }
    .st-emotion-cache-7ym5gk {
    font-size: 16px; /* Adjust the font size to increase button size */
    padding: 10px 20px; /* Adjust padding to increase button size */
    margin: 0 auto; /* Center the button horizontally */
    display: block; /* Ensure the button takes full width */
    }
    .st-emotion-cache-nwtri {
    display: flex;
    justify-content: center;
    }
    
    </style>
    """
    st.markdown(box, unsafe_allow_html=True)

def logo_navbar():
    l="""
    
    """
    