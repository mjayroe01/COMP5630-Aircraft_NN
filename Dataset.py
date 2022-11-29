import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split


def load_dataset(path):
    labels = sorted(os.listdir(path))

    # The above method returns a list of strings rather than ints, so it sorts differently
    # than ints, and MacOS adds a ".DS_Store" file to folders, so we need to
    # ignore that file and create a new list of sorted ints
    tmp = []

    for label in labels:
        if not label.startswith('.'):
            tmp.append(label)

    sorted_int_labels = np.array(tmp).astype('uint8')

    sorted_int_labels.sort()
    #temp line below
    sorted_int_labels = sorted_int_labels[:9]

    X = []
    y = []

    for label in sorted_int_labels:
        for file in os.listdir(os.path.join(path, str(label))):
            if not file.startswith('.'):
                # Read image
                IMG_SIZE = 50
                image = cv2.imread(os.path.join(path, str(label), file))
                new_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

                X.append(new_image)
                y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test).astype('uint8')
