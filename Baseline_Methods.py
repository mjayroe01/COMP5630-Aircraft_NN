import numpy as np
import cv2
from Dataset import load_dataset
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ignore_warnings(category=ConvergenceWarning)

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

# Load the dataset
X_train, X_test, y_train, y_test = load_dataset('dataset')

# Shuffle training dataset
keys = np.array(range(X_train.shape[0]))
np.random.shuffle(keys)
X_train = X_train[keys]
y_train = y_train[keys]

# Scale and reshape samples
X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# Read image to test
image_data = cv2.imread('identify/v22.jpg')

# Resize image
image_data = cv2.resize(image_data, (50, 50))

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5


# Train sklearn MLP
print('-----Begin training sklearn MLP-----')
MLP = MLPClassifier(hidden_layer_sizes=(2000), max_iter=20,activation = 'relu',solver='adam',random_state=42)
MLP.fit(X_train, y_train)

# Get testing accuracy
y_pred = MLP.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
accuracy = cm.trace() / cm.sum()

# Test on a single image
singlePrediction = MLP.predict(image_data)[0]

# Print MLP result
resultString = '-MLP testing accuracy: ' + str(accuracy) + '\n-Prediction on V22 image: ' + str(singlePrediction)
if singlePrediction ==  9:
    resultString += ' = V22 = CORRECT'
else:
    resultString += ' = ' + aircraft_labels[singlePrediction] + ' = INCORRECT'
print(resultString)


# Convert to a binary classification problem for baseline methods
for i,y in enumerate(y_train):
    if y == 9:
        y_train[i] = 1
    else:
        y_train[i] = 0

for i,y in enumerate(y_test):
    if y == 9:
        y_test[i] = 1
    else:
        y_test[i] = 0


# Train sklearn Logistic Regression model
print('-----Begin training sklearn Logistic Regression-----')
LR = LogisticRegression()
LR.fit(X_train, y_train)

# Get testing accuracy
accuracy = LR.score(X_test, y_test)

# Test on a single image
singlePrediction = LR.predict(image_data)[0]

# Print logistic regression result
resultString = '-Logistic Regression testing accuracy: ' + str(accuracy) + '\n-Prediction on V22 image: ' + str(singlePrediction)
if singlePrediction ==  1:
    resultString += ' = V22 = CORRECT'
else:
    resultString += ' = not V22 = INCORRECT'
print(resultString)


# Train sklearn SVM model
print('-----Begin training sklearn SVM-----')
SVM = SVC(kernel='linear')
SVM.fit(X_train, y_train)

# Get testing accuracy
accuracy = SVM.score(X_test, y_test)

# Test on a single image
singlePrediction = SVM.predict(image_data)[0]

# Print SVM result
resultString = '-SVM testing accuracy: ' + str(accuracy) + '\n-Prediction on V22 image: ' + str(singlePrediction)
if singlePrediction ==  1:
    resultString += ' = V22 = CORRECT'
else:
    resultString += ' = not V22 = INCORRECT'
print(resultString)

