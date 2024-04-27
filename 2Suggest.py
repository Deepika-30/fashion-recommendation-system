import streamlit as st
from streamlit_option_menu import option_menu
import random
import streamlit.components.v1 as components #1
from streamlit_lottie import st_lottie  #4
import json
import requests #4
import tensorflow
from keras.src.applications.convnext import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import pickle
import os
from PIL import Image
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from st_clickable_images import clickable_images
import front_designing

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
#7 setting page configuration
st.set_page_config(page_title="Recommend similar products", page_icon=":magnifying_glass_tilted_left:", layout="wide")
front_designing.side_nav_bar()

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 80px !important;  # Set the width to your desired value
    }
</style>
""", unsafe_allow_html=True)


#7
#9 setting variables for json
lottie_choosing=load_lottiefile("./json_lottie/choosing.json")
lottie_upload=load_lottiefile("./json_lottie/upload_image.json")
lottie_timer=load_lottiefile("./json_lottie/timer.json")
lottie_guide=load_lottiefile("./json_lottie/guide.json")
lottie_result=load_lottiefile("./json_lottie/result.json")
#--header section---
#8
#path_to_html = "./front_recommendation.html" #2
#with open(path_to_html,'r') as f:
#    html_data = f.read()
#with st.container():
    #Show in webpage
    #st.header("Show an external HTML")
#    st.components.v1.html(html_data)#2

# with st.sidebar:
#     selected= option_menu(
#         menu_title="Main Menu",
#         options=["Home", "FAQ", "About Us"],
#         icons=["house-door", "chat-left-quote", "person-arms-up"]
#     )
# if selected=="Home":
#     pass
with st.container():
    st.subheader("Hi, I am Vihaan :wave:")
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Exhausted from the endless search for a dress that matches one from your favorite image?")
        st.write("Welcome to our innovative platform where the magic of AI helps you discover products that match the style of your uploaded image. Simply upload a picture, and let our intelligent system curate a selection of similar items tailored just for you. Dive into a personalized shopping experience like no other! Our platform offers a seamless solution, effortlessly connecting you with similar styles at the click of a button. Experience the ease of finding your desired dress without the hassle.")

    with right_column:
        st_lottie(lottie_choosing, height=400, key="choosing")
#8
#3
#'''page_by_image="""
#<style>
#</style>
#"""'''#3
#st.title('Fashion Recommender system')  #6
#st.subheader('Welcome to fashion hub')  #6
#"""def load_lottieurl(url:str): #5
#    r=requests.get(url)
#    if r.status_code!=200:
#        return None
#    return r.json()
#fashion1=load_lottieurl()"""



feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])




def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def extract_features(img_path, model, axis=None):
    img = image.load_img(img_path,target_size =(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis ==0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

with st.container():
    st.write("---")
    st.subheader('“Snap a pic, unlock inspiration! Discover similar styles just for you.”')
    uploaded_file = st.file_uploader("choose an image")
css = '''
<style>
    [data-testid='stFileUploader'] {
        width: 20;
        height:20;
    }
    [data-testid='stFileUploader'] section {
        padding: 500;
        position: center;
    }
    [data-testid='stFileUploader'] section > input + div {
        display: block;
    }
    [data-testid='stFileUploader'] section + div {
        float: none;
        padding-top: 100;
    }

</style>
'''


front_designing.drop_box()
st.markdown(css, unsafe_allow_html=True)

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image, width=150)
        feature = extract_features(os.path.join("upload", uploaded_file.name), model )
        indices = recommend(feature,feature_list)
        
        with st.container():
            col1,col2,col3,col4,col5,col6,col7,col8,col9 = st.columns([0.3,1,0.3,1,0.3,1,0.3,1,0.3])

            with col2:
                st.image(filenames[indices[0][0]], width=150)

            with col4:
                st.image(filenames[indices[0][1]], width=150)

            with col6:
                st.image(filenames[indices[0][2]], width=150)

            with col8:
                st.image(filenames[indices[0][3]], width = 150)
        with st.container():
            col1,col2,col3,col4,col5,col6,col7,col8,col9 = st.columns([0.3,1,0.3,1,0.3,1,0.3,1,0.3])
            with col2:
                st.image(filenames[indices[0][4]], width = 150)
            with col4:
                st.image(filenames[indices[0][5]], width = 150)
            with col6:
                st.image(filenames[indices[0][6]], width = 150)
            with col8:
                st.image(filenames[indices[0][7]], width = 150)
    else:
        st.header("some error in file upload")

    ##########    trendy item

    # st.subheader("Check trendy items")


    # dataset_dir = 'images'

    # # Get list of image file paths in the dataset directory
    # image_paths = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if
    #                file.endswith(('.jpg', '.jpeg', '.png'))]

    # # Shuffle the list of image paths
    # random.shuffle(image_paths)

    # # Display a random image

    # num_images_to_display = 10  # Change this value to display a different number of images

    # num_columns = 3  # Number of columns to display images side by side

    # # Create Streamlit columns for each row
    # for i in range(0, num_images_to_display, num_columns):
    #     columns = st.columns(num_columns)
    #     for j in range(num_columns):
    #         index = i + j
    #         if index < len(image_paths):
    #             columns[j].image(image_paths[index], use_column_width=False,width=150)
    ###################### trendy end
with st.container():
    st.write("---")
    with st.container():
        left_column,right_column=st.columns([2,1])
        with left_column:
            st.header("Guide: How to use it ")
            st.write("Experience effortless navigation with our user-friendly website. Simply follow these straightforward steps to discover recommendations and find products that align with your preferences.")
        with right_column:
            st_lottie(lottie_guide, height=150, width=300, key="guide")
    with st.container():
        
        left_column,right_column=st.columns([2,1])
        with left_column:
            st.subheader("Step 1: Upload a file")
            st.write("To find products similar to your interests, you can either drag and drop an image into the designated drop box or use the ‘Browse’ button to select a file from your device. Ensure the image accurately represents the item you’re searching for to receive the best possible matches.")
        with right_column:
            st_lottie(lottie_upload, height=150, width=300, key="upload")

    with st.container():
        left_column1,right_column1=st.columns(2)
        with left_column:
            st.subheader("Step 2 : Wait for few moments to get the results")
            st.write("The recommendation engine is swiftly curating the best product suggestions for you. This may take a moment, but rest assured, the results will be well worth the wait. Your patience is appreciated.")
        with right_column:
            st_lottie(lottie_timer, height=150, width=300, key="timer")

    with st.container():
        left_column1,right_column1=st.columns(2)
        with left_column:
            st.subheader("Step 3 : Get Results")
            st.write("I hope you find these suggestions to your liking and that they lead you to the results you’ve been seeking. Please feel free to return anytime for more assistance. Your feedback is greatly appreciated, and I hope I’ve been of help to you.")
        with right_column:
            st_lottie(lottie_result, height=100, width=300, key="result")

    