import numpy as np
from Dataset1 import create_dataset
from Dataset2 import load_dataset
from Model import Model
from Layers import Layer_Dense, Layer_Dropout
from Activation_Functions import Activation_ReLU, Activation_Softmax
from Metric_Functions import Loss_CategoricalCrossEntropy, Accuracy_Categorical
from Optimizers import Optimizer_Adam

X_train, X_test, y_train, y_test = load_dataset('crop', 'archive2')

# Shuffle training dataset
keys = np.array(range(X_train.shape[0]))
np.random.shuffle(keys)
X_train = X_train[keys]
y_train = y_train[keys]

# Scale and reshape samples
X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# Instantiate our model
model = Model()

# Add layers
model.add(Layer_Dense(X_train.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.25))
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.25))
model.add(Layer_Dense(128, 40))
model.add(Activation_Softmax())

# Set loss, accuracy and optimizer
model.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

# Finalize model
model.finalize()

# Train!
model.train(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=40, print_every=100)

# Evaluate!
model.evaluate(X_test, y_test)
