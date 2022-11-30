import numpy as np
from Dataset import load_dataset
from Model import Model
from Layers import Layer_Dense, Layer_Dropout
from Activation_Functions import Activation_ReLU, Activation_Softmax
from Metric_Functions import Loss_CategoricalCrossEntropy, Accuracy_Categorical
from Optimizers import Optimizer_Adam

X_train, X_test, y_train, y_test = load_dataset('dataset')

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
inputLayer = X_train.shape[1]
layerNum1 = 2000
outputLayer = 10
print(X_train.shape)
print("input layer: ",inputLayer,
      "\nhidden layer 1: ",layerNum1,
      "\noutput layer: ",outputLayer)
model.add(Layer_Dense(inputLayer, inputLayer))
model.add(Activation_ReLU())
model.add(Layer_Dense(inputLayer, layerNum1))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.3))
model.add(Layer_Dense(layerNum1, outputLayer))
model.add(Activation_Softmax())

# Set loss, accuracy and optimizer
model.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

# Finalize model
model.finalize()

# Load a saved model to continue training
# If one wants to load a previous model, comment out
# all lines above that contain 'model' and instantiate
# the model with the load function.
#model = model.load('military_aircraft_trained_mlp.model')

# Train!
model.train(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=40, print_every=100)

# Evaluate!
model.evaluate(X_test, y_test)

# Save model
model.save('military_aircraft_trained_mlp.model')
