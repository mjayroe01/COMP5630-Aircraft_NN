import numpy as np
from Dataset import create_dataset
from Model import Model
from Layers import Layer_Dense
from Activation_Functions import Activation_ReLU, Activation_Softmax
from Metric_Functions import Loss_CategoricalCrossEntropy, Accuracy_Categorical
from Optimizers import Optimizer_Adam

X, y, X_test, y_test = create_dataset('archive')

# Shuffle training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# Instantiate our model
model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
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
model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=40, print_every=6)

# Evaluate!
model.evaluate(X_test, y_test)
