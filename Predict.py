import cv2
import numpy as np
from Model import Model

aircraft_labels = {
    0: 'A10',
    1: 'A400M',
    2: 'AG600',
    3: 'AV8B',
    4: 'B1',
    5: 'B2',
    6: 'B52',
    7: 'Be200',
    8: 'C5',
    9: 'C17'
}

# Read image
image_data = cv2.imread('identify/a10.png')

# Resize image
image_data = cv2.resize(image_data, (50, 50))

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load model
model = Model.load('military_aircraft_mlp.model')

# Predict using image
confidences = model.predict(image_data)

# Get prediction instead of confidences
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from index
prediction = aircraft_labels[predictions[0]]

print(prediction)
