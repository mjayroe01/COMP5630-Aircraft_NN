import numpy as np
import os
import cv2

def load_dataset(dataset, path):
    # Scan all directories and make list of labels
    labels = sorted(os.listdir(os.path.join(path, dataset)))

    X = []
    y = []

    for label in labels:
        if not label.startswith('.'):
            # For each image in folder
            for file in os.listdir(os.path.join(path, dataset, label)):
                if not file.startswith('.'):
                    # Read image
                    IMG_SIZE = 200
                    image = cv2.imread(os.path.join(path, dataset, label, file))
                    new_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

                    # Append imagine and label to lists
                    X.append(new_image)
                    y.append(label)

    return np.array(X), np.array(y).astype('uint8')

def create_dataset(path):
    # Load sets separately
    X, y = load_dataset('train', path)
    X_test, y_test = load_dataset('test', path)

    return X, y, X_test, y_test

