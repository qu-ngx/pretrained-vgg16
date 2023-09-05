import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

# Load Keras' VGG16 model (pretrained)
model = vgg16.VGG16()

# Load the image file, resizing it to 224x244 pix
img = image.load_img("bay.jpg", target_size=(224, 224))

# Convert the img to a np arr
x = image.img_to_array(img)

# Add a fourth dim (Since Keras expects a list of images)
x = np.expand_dims(x, axis=0)

# Normalize the input image's pix values
x = vgg16.preprocess_input(x)

# Run the image through deep neural network to make a prediction
predictions = model.predict(x)

# Look up the names of the predicted classes
predicted_classes = vgg16.decode_predictions(predictions, top=9)

for imagenet_id, name, likelihood in predicted_classes[0]:
    print("Prediction: {} - {:2f}".format(name, likelihood))
