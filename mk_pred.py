from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np  
from keras.applications import vgg16

# Load the json file that contains the model's struct
f = Path("model_structure.json")
model_structure = f.read_text()

# Recreate the Kernel model object from json data
model = model_from_json(model_structure)

# Re-load the model's trained weights 
model.load_weights("model_weights.h5")

# Load an image file to test, resizing it to 64 x 64
img = image.load_img("dog.png", target_size=(64, 64))

# Convert the image to a numpy array
image_arr = image.img_to_array(img)
print(image_arr.shape)

# Add a fourth dimension to the image 
images = np.expand_dims(image_arr, axis=0)
print(images.shape)

# Normalize the data 
images = vgg16.preprocess_input(images)
print(images.shape)

# Use the pre-trained neuraL network to extract features from test image
feature_extraction_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
features = feature_extraction_model.predict(images)

# Given the extracted features, make a final prediction using our own model
results = model.predict(features)

# Check with only the first result, since we are testing 1 image
single_result = results[0][0]

# Print the result:
print("Likelihood of the image containing a dog: {}%".format(int(single_result * 100)))

