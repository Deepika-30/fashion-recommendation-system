import streamlit as st
from streamlit_lottie import st_lottie
import json
import requests
import front_designing
import tensorflow as tf
import itertools
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


st.set_page_config(page_title="Vihaan-Recommend", page_icon=":speech_balloon:", layout="wide")
front_designing.side_nav_bar()

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 80px !important;  # Set the width to your desired value
    }
</style>
""", unsafe_allow_html=True)

lottie_upload=load_lottiefile("./json_lottie/upload_image.json")
lottie_timer=load_lottiefile("./json_lottie/timer.json")
lottie_guide=load_lottiefile("./json_lottie/guide.json")
lottie_result=load_lottiefile("./json_lottie/result.json")

with st.container():
    st.subheader("Hi, I am Vihaan :wave:")
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Stuck in a Wardrobe Dilemma? :exploding_head:")
        st.write("Do you find yourself staring at a closet full of clothes, yet unable to decide what to wear? Maybe you’ve got new additions that you can’t bear to part with, but the struggle to choose an outfit feels real. Fear not! We’re here to assist you in curating the perfect ensemble. Let’s turn your fashion conundrum into a stylish solution! ")
    with right_column:
        lottie_stare=load_lottiefile("./json_lottie/stare.json")
        st_lottie(lottie_stare, height=300, key="stare")


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
    
with st.container():
    st.write("---")
    st.subheader('“Discover the power of your style! Drop images and let our experts curate the perfect outfit for you.”')
#     uploaded_file = st.file_uploader("choose an image")
# css = '''
# <style>
#     [data-testid='stFileUploader'] {
#         width: 20;
#         height:20;
#     }
#     [data-testid='stFileUploader'] section {
#         padding: 500;
#         position: center;
#     }
#     [data-testid='stFileUploader'] section > input + div {
#         display: block;
#     }
#     [data-testid='stFileUploader'] section + div {
#         float: none;
#         padding-top: 100;
#     }

# </style>
# '''


# front_designing.drop_box()
# st.markdown(css, unsafe_allow_html=True)

#ashish paste your code here below this comment
model_path = 'D:/codes/project/fashionproject/classification/wear.h5'
model = tf.keras.models.load_model(model_path)


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

#col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
front_designing.drop_box()
predictions_list = []

if uploaded_files:



    upper_predictions = []
    lower_predictions = []
    footwear_predictions = []
    #
    for uploaded_file in uploaded_files:
        pic = Image.open(uploaded_file)
        st.image(pic, caption='Uploaded Image', use_column_width=False, width=150)
# yha p edit kr lena upper p



        # Preprocess the image and make predictions
        user_image = preprocess_image(uploaded_file)
        prediction = model.predict(user_image)
        predicted_class_index = np.argmax(prediction)



        def get_class_label(class_index):
            class_labels = {0: "footwear",
                            1: "bottomwear",
                            2: "upperwear"}
            return class_labels[class_index]



        predicted_class_label = get_class_label(predicted_class_index)
        st.write("image:", predicted_class_label)


        if predicted_class_label == 'upperwear':
            upper_predictions.append(uploaded_file)
        elif predicted_class_label == 'bottomwear':
            lower_predictions.append(uploaded_file)
        elif predicted_class_label == 'footwear':
            footwear_predictions.append(uploaded_file)


    #st.write("Predicted Upperwear images:", [file.name for file in upper_predictions])
    #st.write("Predicted Lowerwear images:", [file.name for file in lower_predictions])
    #st.write("Predicted Footwear images:", [file.name for file in footwear_predictions])

    combinations = list(itertools.product(upper_predictions, lower_predictions, footwear_predictions))

    no_of_rows = len(combinations) // 3 + (1 if len(combinations) % 3 != 0 else 0)

    if combinations:
        st.write("Recommended Outfit Combinations:")
        for row_i in range(no_of_rows):
            left_column = st.columns(3)
            for i in range(0, 3):
                combo_index = row_i * 3 + i
                if combo_index < len(combinations):
                    combo = combinations[combo_index]
                    with left_column[i]:
                        st.header(f"Outfit {combo_index + 1}:")
                        st.image(combo[0], caption='Upperwear', use_column_width=False, width=150)
                        st.image(combo[1], caption='Lowerwear', use_column_width=False, width=150)
                        st.image(combo[2], caption='Footwear', use_column_width=False, width=150)

            st.write("\n")
           # st.write("\n")
    else:
        st.write("No outfit combinations available. Please upload more images.")



#above this comment

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
            st.subheader("Step 1: Upload your files")
            st.write("Upload or drop your files here, and let our smart system create the perfect clothing combinations for you. Say goodbye to shopping stress and that feeling of not having enough outfits")
        with right_column:
            st_lottie(lottie_upload, height=150, width=300, key="upload")

    with st.container():
        left_column1,right_column1=st.columns(2)
        with left_column:
            st.subheader("Step 2 : Wait for few moments to get the results")
            st.write("Please bear with us for a moment while we curate the perfect ensemble for you. We’ll ensure you’re dressed to impress. Just a few moments, and your stylish outfit will be ready!  ")
        with right_column:
            st_lottie(lottie_timer, height=150, width=300, key="timer")

    with st.container():
        left_column1,right_column1=st.columns(2)
        with left_column:
            st.subheader("Step 3 : Get Results")
            st.write("Your anticipation has come to an end! Behold, the results are here, tailored just for you.  We’d love to hear your thoughts, so don’t forget to share your feedback. Happy styling!")
        with right_column:
            st_lottie(lottie_result, height=100, width=300, key="result")