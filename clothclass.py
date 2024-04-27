import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image
import itertools

st.header("Classify the items")
para = ("Elevate your wardrobe with timeless elegance and modern flair. Embrace the artistry of fashion as you navigate through a spectrum of styles, from chic sophistication to casual comfort. Let your ensemble speak volumes about your unique personality and taste, as you effortlessly blend comfort with couture. Experience the magic of clothing that transcends trends and resonates with your individuality."
        " &mdash;\
            :shirt::jeans::T-shirt::shoe:")
st.markdown(para)

model_path = 'D:/fashionproject/classification/wear.h5'
model = tf.keras.models.load_model(model_path)


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

#col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
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

'''
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
st.header("Classify the item")
para = "Elevate your wardrobe with timeless elegance and modern flair. Embrace the artistry of fashion as you navigate through a spectrum of styles, from chic sophistication to casual comfort. Let your ensemble speak volumes about your unique personality and taste, as you effortlessly blend comfort with couture. Experience the magic of clothing that transcends trends and resonates with your individuality. &mdash;\
            :shirt::jeans::T-shirt::shoe:"
st.markdown(para)

model_path = 'D:/fashionproject/classification/wear.h5'
model = tf.keras.models.load_model(model_path)


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


user_image_path = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


if user_image_path is not None:
    pic = Image.open(user_image_path)
    st.image(pic, caption='Uploaded Image', use_column_width=False,  width = 100)
    user_image = preprocess_image(user_image_path)

    predictions = model.predict(user_image)
    predicted_class_index = np.argmax(predictions)


    def get_class_label(class_index):
        class_labels = {0: "footwear",
                        1: "bottomwear",
                        2: "upperwear"}
        return class_labels[class_index]

    predicted_class_label = get_class_label(predicted_class_index)


    st.header(predicted_class_label)
'''
'''import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image
import itertools

st.header("Classify the items")
para = "Elevate your wardrobe with timeless elegance and modern flair. Embrace the artistry of fashion as you navigate through a spectrum of styles, from chic sophistication to casual comfort. Let your ensemble speak volumes about your unique personality and taste, as you effortlessly blend comfort with couture. Experience the magic of clothing that transcends trends and resonates with your individuality."
st.markdown(para)

model_path = 'D:/fashionproject/classification/wear.h5'
model = tf.keras.models.load_model(model_path)


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
predictions_list = []
if uploaded_files:

    upperwear_predictions = []
    lowerwear_predictions = []
    footwear_predictions = []

    for uploaded_file in uploaded_files:
        pic = Image.open(uploaded_file)
        st.image(pic, caption='Uploaded Image', use_column_width=False,  width = 100)

        # Preprocess the image and make predictions
        user_image = preprocess_image(uploaded_file)
        prediction = model.predict(user_image)
        predicted_class_index = np.argmax(prediction)


        # Function to get the class label
        def get_class_label(class_index):
            class_labels = {0: "footwear",
                            1: "bottomwear",
                            2: "upperwear"}
            return class_labels[class_index]


        # Get the predicted class label
        predicted_class_label = get_class_label(predicted_class_index)

        # Store the filenames in corresponding lists based on predicted classes
        if predicted_class_label == 'upperwear':
            upperwear_predictions.append(uploaded_file.name if uploaded_file.name else "Image")
        elif predicted_class_label == 'bottomwear':
            lowerwear_predictions.append(uploaded_file.name if uploaded_file.name else "Image")
        elif predicted_class_label == 'footwear':
            footwear_predictions.append(uploaded_file.name if uploaded_file.name else "Image")

    # Display the predicted filenames for each category
    st.write("Predicted Upperwear images:", upperwear_predictions)
    st.write("Predicted Lowerwear images:", lowerwear_predictions)
    st.write("Predicted Footwear images:", footwear_predictions)


    # Get all possible combinations of one upperwear, one lowerwear, and one footwear
    combinations = list(itertools.product(upperwear_predictions, lowerwear_predictions, footwear_predictions))

    # Display the combinations to the user
    if combinations:
        st.write("Recommended Outfit Combinations:")
        for idx, combo in enumerate(combinations):
            st.write(f"Outfit {idx + 1}:")
            st.write("Upperwear:", combo[0])
            st.write("Lowerwear:", combo[1])
            st.write("Footwear:", combo[2])

            with open(combo[0], "rb") as upperwear_file:
                st.image(upperwear_file, caption='Upperwear', use_column_width=False,  width = 100)

            with open(combo[1], "rb") as lowerwear_file:
                st.image(lowerwear_file, caption='Lowerwear', use_column_width=False,  width = 100)

            with open(combo[2], "rb") as footwear_file:
                st.image(footwear_file, caption='Footwear', use_column_width=False,  width = 100)

            st.write("\n")
            st.write("\n")

            st.image(combo[0], caption='Upperwear', use_column_width=False,  width = 100)
            st.image(combo[1], caption='Lowerwear', use_column_width=False,  width = 100)
            st.image(combo[2], caption='Footwear', use_column_width=False,  width = 100)
            st.write("\n")
            st.write("\n")
    else:
        st.write("No outfit combinations available. Please upload more images.")
'''

'''predictions_list
for i in range(predictions_list):
    if b[i] == "upperwear":
        st.write("chose")

    elif b[i]=="bottomwear":
        st.write("bottom")
    else:
        st.write("foot")'''

'''import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image

st.header("Classify the items")
para = "Elevate your wardrobe with timeless elegance and modern flair. Embrace the artistry of fashion as you navigate through a spectrum of styles, from chic sophistication to casual comfort. Let your ensemble speak volumes about your unique personality and taste, as you effortlessly blend comfort with couture. Experience the magic of clothing that transcends trends and resonates with your individuality."
st.markdown(para)

model_path = 'D:/fashionproject/classification/wear.h5'
model = tf.keras.models.load_model(model_path)


def preprocess_image(image_path, target_size=(28, 28)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    predictions = []
    image_names = []

    for i, uploaded_file in enumerate(uploaded_files):
        pic = Image.open(uploaded_file)
        st.image(pic, caption=f'Uploaded Image {i+1}', use_column_width=False,  width = 100)
        user_image = preprocess_image(uploaded_file)

        prediction = model.predict(user_image)
        predicted_class_index = np.argmax(prediction)

        def get_class_label(class_index):
            class_labels = {0: "footwear",
                            1: "bottomwear",
                            2: "upperwear"}
            return class_labels[class_index]

        predicted_class_label = get_class_label(predicted_class_index)

        predictions.append(predicted_class_label)
        image_names.append(uploaded_file.name if uploaded_file.name else f"Image {i+1}")

    st.write("Predicted class labels:", predictions)
    st.write("Image names:", image_names)'''
'''import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image

st.header("Classify the items")
para = "Elevate your wardrobe with timeless elegance and modern flair. Embrace the artistry of fashion as you navigate through a spectrum of styles, from chic sophistication to casual comfort. Let your ensemble speak volumes about your unique personality and taste, as you effortlessly blend comfort with couture. Experience the magic of clothing that transcends trends and resonates with your individuality."
st.markdown(para)

model_path = 'D:/fashionproject/classification/wear.h5'
model = tf.keras.models.load_model(model_path)


def preprocess_image(image_path, target_size=(100, 100)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    upperwear_predictions = []
    lowerwear_predictions = []
    footwear_predictions = []

    for uploaded_file in uploaded_files:
        pic = Image.open(uploaded_file)
        st.image(pic.resize((100, 100)), caption='Uploaded Image', use_column_width=False,  width = 100)
        user_image = preprocess_image(uploaded_file)

        prediction = model.predict(user_image)
        predicted_class_index = np.argmax(prediction)

        def get_class_label(class_index):
            class_labels = {0: "footwear",
                            1: "bottomwear",
                            2: "upperwear"}
            return class_labels[class_index]

        predicted_class_label = get_class_label(predicted_class_index)

        if predicted_class_label == 'upperwear':
            upperwear_predictions.append(uploaded_file.name if uploaded_file.name else "Image")
        elif predicted_class_label == 'bottomwear':
            lowerwear_predictions.append(uploaded_file.name if uploaded_file.name else "Image")
        elif predicted_class_label == 'footwear':
            footwear_predictions.append(uploaded_file.name if uploaded_file.name else "Image")

    st.write("Predicted Upperwear images:", upperwear_predictions)
    st.write("Predicted Lowerwear images:", lowerwear_predictions)
    st.write("Predicted Footwear images:", footwear_predictions)
'''
'''import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image

# Header and introductory paragraph
st.header("Classify the items")
para = "Elevate your wardrobe with timeless elegance and modern flair. Embrace the artistry of fashion as you navigate through a spectrum of styles, from chic sophistication to casual comfort. Let your ensemble speak volumes about your unique personality and taste, as you effortlessly blend comfort with couture. Experience the magic of clothing that transcends trends and resonates with your individuality."
st.markdown(para)

# Load the pre-trained model
model_path = 'D:/fashionproject/classification/wear.h5'
model = tf.keras.models.load_model(model_path)


# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# File uploader for multiple images
uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    upperwear_predictions = []
    lowerwear_predictions = []
    footwear_predictions = []

    for uploaded_file in uploaded_files:
        pic = Image.open(uploaded_file)
        # Display the uploaded image
        st.image(pic, caption='Uploaded Image', use_column_width=False,  width = 100)

        # Preprocess the image and make predictions
        user_image = preprocess_image(uploaded_file)
        prediction = model.predict(user_image)
        predicted_class_index = np.argmax(prediction)


        # Function to get the class label
        def get_class_label(class_index):
            class_labels = {0: "footwear",
                            1: "bottomwear",
                            2: "upperwear"}
            return class_labels[class_index]


        # Get the predicted class label
        predicted_class_label = get_class_label(predicted_class_index)

        # Store the filenames in corresponding lists based on predicted classes
        if predicted_class_label == 'upperwear':
            upperwear_predictions.append(uploaded_file.name if uploaded_file.name else "Image")
        elif predicted_class_label == 'bottomwear':
            lowerwear_predictions.append(uploaded_file.name if uploaded_file.name else "Image")
        elif predicted_class_label == 'footwear':
            footwear_predictions.append(uploaded_file.name if uploaded_file.name else "Image")

    # Display the predicted filenames for each category
    st.write("Predicted Upperwear images:", upperwear_predictions)
    st.write("Predicted Lowerwear images:", lowerwear_predictions)
    st.write("Predicted Footwear images:", footwear_predictions)
    import itertools

    # Get all possible combinations of one upperwear, one lowerwear, and one footwear
    combinations = list(itertools.product(upperwear_predictions, lowerwear_predictions, footwear_predictions))

    # Display the combinations to the user
    if combinations:
        st.write("Recommended Outfit Combinations:")
        for idx, combo in enumerate(combinations):
            st.write(f"Outfit {idx + 1}:")
            st.write("Upperwear:", combo[0])
            st.write("Lowerwear:", combo[1])
            st.write("Footwear:", combo[2])
            st.write("\n")
    else:
        st.write("No outfit combinations available. Please upload more images.")'''

'''import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image
import itertools

st.header("Classify the items")
para = "Elevate your wardrobe with timeless elegance and modern flair. Embrace the artistry of fashion as you navigate through a spectrum of styles, from chic sophistication to casual comfort. Let your ensemble speak volumes about your unique personality and taste, as you effortlessly blend comfort with couture. Experience the magic of clothing that transcends trends and resonates with your individuality."
st.markdown(para)

model_path = 'D:/fashionproject/classification/wear.h5'
model = tf.keras.models.load_model(model_path)


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
predictions_list = []
if uploaded_files:

    upperwear_predictions = []
    lowerwear_predictions = []
    footwear_predictions = []

    for uploaded_file in uploaded_files:
        pic = Image.open(uploaded_file)
        st.image(pic, caption='Uploaded Image', use_column_width=False,  width = 100)

        # Preprocess the image and make predictions
        user_image = preprocess_image(uploaded_file)
        prediction = model.predict(user_image)
        predicted_class_index = np.argmax(prediction)


        # Function to get the class label
        def get_class_label(class_index):
            class_labels = {0: "footwear",
                            1: "bottomwear",
                            2: "upperwear"}
            return class_labels[class_index]


        # Get the predicted class label
        predicted_class_label = get_class_label(predicted_class_index)

        # Store the filenames in corresponding lists based on predicted classes
        if predicted_class_label == 'upperwear':
            upperwear_predictions.append(pic)
        elif predicted_class_label == 'bottomwear':
            lowerwear_predictions.append(pic)
        elif predicted_class_label == 'footwear':
            footwear_predictions.append(pic)

    # Display the predicted filenames for each category
    st.write("Predicted Upperwear images:", [file.name for file in upperwear_predictions])
    st.write("Predicted Lowerwear images:", [file.name for file in lowerwear_predictions])
    st.write("Predicted Footwear images:", [file.name for file in footwear_predictions])

    # Get all possible combinations of one upperwear, one lowerwear, and one footwear
    combinations = list(itertools.product(upperwear_predictions, lowerwear_predictions, footwear_predictions))

    # Display the combinations to the user
    if combinations:
        st.write("Recommended Outfit Combinations:")
        for idx, combo in enumerate(combinations):
            st.write(f"Outfit {idx + 1}:")
            st.write("Upperwear:", combo[0].name)
            st.write("Lowerwear:", combo[1].name)
            st.write("Footwear:", combo[2].name)

            resized_upperwear_img = combo[0].resize((100, 100))
            resized_lowerwear_img = combo[1].resize((100, 100))
            resized_footwear_img = combo[2].resize((100, 100))

            st.image(resized_upperwear_img, caption='Upperwear', use_column_width=False,  width = 100)
            st.image(resized_lowerwear_img, caption='Lowerwear', use_column_width=False,  width = 100)
            st.image(resized_footwear_img, caption='Footwear', use_column_width=False,  width = 100)

            st.write("\n")
            st.write("\n")
    else:
        st.write("No outfit combinations available. Please upload more images.")'''
'''
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image
import itertools

st.header("Classify the items")
para = "Elevate your wardrobe with timeless elegance and modern flair. Embrace the artistry of fashion as you navigate through a spectrum of styles, from chic sophistication to casual comfort. Let your ensemble speak volumes about your unique personality and taste, as you effortlessly blend comfort with couture. Experience the magic of clothing that transcends trends and resonates with your individuality."
st.markdown(para)

model_path = 'D:/fashionproject/classification/wear.h5'
model = tf.keras.models.load_model(model_path)


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
predictions_list = []
if uploaded_files:

    upperwear_predictions = []
    lowerwear_predictions = []
    footwear_predictions = []

    for uploaded_file in uploaded_files:
        pic = Image.open(uploaded_file)
        st.image(pic, caption='Uploaded Image', use_column_width=False,  width = 100)

        # Preprocess the image and make predictions
        user_image = preprocess_image(uploaded_file)
        prediction = model.predict(user_image)
        predicted_class_index = np.argmax(prediction)


        # Function to get the class label
        def get_class_label(class_index):
            class_labels = {0: "footwear",
                            1: "bottomwear",
                            2: "upperwear"}
            return class_labels[class_index]


        # Get the predicted class label
        predicted_class_label = get_class_label(predicted_class_index)

        # Store the filenames in corresponding lists based on predicted classes
        if predicted_class_label == 'upperwear':
            upperwear_predictions.append(uploaded_file.name)
        elif predicted_class_label == 'bottomwear':
            lowerwear_predictions.append(uploaded_file.name)
        elif predicted_class_label == 'footwear':
            footwear_predictions.append(uploaded_file.name)

    # Display the predicted filenames for each category
    st.write("Predicted Upperwear images:", upperwear_predictions)
    st.write("Predicted Lowerwear images:", lowerwear_predictions)
    st.write("Predicted Footwear images:", footwear_predictions)

    # Get all possible combinations of one upperwear, one lowerwear, and one footwear
    combinations = list(itertools.product(upperwear_predictions, lowerwear_predictions, footwear_predictions))

    # Display the combinations to the user
    if combinations:
        st.write("Recommended Outfit Combinations:")
        for idx, combo in enumerate(combinations):
            st.write(f"Outfit {idx + 1}:")
            st.write("Upperwear:", combo[0])
            st.write("Lowerwear:", combo[1])
            st.write("Footwear:", combo[2])

            resized_upperwear_img = Image.open(combo[0]).resize((100, 100))
            resized_lowerwear_img = Image.open(combo[1]).resize((100, 100))
            resized_footwear_img = Image.open(combo[2]).resize((100, 100))

            st.image(resized_upperwear_img, caption='Upperwear', use_column_width=False,  width = 100)
            st.image(resized_lowerwear_img, caption='Lowerwear', use_column_width=False,  width = 100)
            st.image(resized_footwear_img, caption='Footwear', use_column_width=False,  width = 100)

            st.write("\n")
            st.write("\n")
    else:
        st.write("No outfit combinations available. Please upload more images.")''
'''