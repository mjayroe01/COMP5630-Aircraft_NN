import cv2
import numpy as np
from Model import Model

aircraft_labels = {
    0: 'AV8B',
    1: 'B1',
    2: 'C130',
    3: 'F15',
    4: 'F16',
    5: 'F18',
    6: 'F22',
    7: 'F35',
    8: 'Tornado',
    9: 'V22'
}

# Asks used what image to predict (default is v22.jpg)
image_name = input('Enter image name (leave blank for v22.jpg): ')
image_path = 'identify/'
if '.jpg' in image_name:
    image_path += image_name
else:
    image_path += 'v22.jpg'

# Read image
print('Predicting',image_path)
image_data = cv2.imread(image_path)

# Resize image
image_data = cv2.resize(image_data, (50, 50))

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load model
model = Model.load('military_aircraft_trained_mlp.model')

# Predict using image
confidences = model.predict(image_data)

# Get prediction instead of confidences
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from index
prediction = aircraft_labels[predictions[0]]

print(prediction)


