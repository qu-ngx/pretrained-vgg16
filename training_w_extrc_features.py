from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import joblib

# Load dts
x_train = joblib.load("x_train.dat")
y_train = joblib.load("y_train.dat")

# Create a model and add layers
model = Sequential()

# The first layer 
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=10,
    shuffle=True
)

# Save Neural network structure
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# Save the neural network's trained weights
model.save_weights("model_weights.h5")