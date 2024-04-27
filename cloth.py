import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('classification.h5')
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# Class labels for Fashion MNIST dataset
class_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


image_path = 'th.jpeg'

img_array = load_and_preprocess_image(image_path)

predictions = model.predict(img_array)

predicted_class = np.argmax(predictions)

class_name = class_labels[predicted_class]

img = image.load_img(image_path, target_size=(150, 150))
plt.imshow(img)
plt.axis('off')
plt.title('Predicted class: ' + class_name)
plt.show()