import tensorflow as tf # pip tensorflow in cli window to install the lib.
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained MobileNetV2 model + higher level layers
model = MobileNetV2(weights='imagenet')

# Load and preprocess the image
img_path = r"C:\Users\rashm\OneDrive\Pictures\Ram\OIP.jpg" # change image size
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)


predictions = model.predict(img_array)  # make predictions
predicted_classes = decode_predictions(predictions, top=3)[0]  


plt.figure(figsize=(10 ,10))
plt.imshow(img)
plt.axis('off')
plt.title("Predictions:")
for i, (imagenet_id, label, score) in enumerate(predicted_classes):
    plt.text(10, 10 + i * 30, f"{label}: {score:.2f}", color='white', backgroundcolor='black', fontsize=12)

plt.show()
