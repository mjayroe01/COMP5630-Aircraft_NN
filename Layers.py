import numpy as np

class Layer_Dense:
    # Initialize layer
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs, train):
        # Remember input values
        self.inputs = inputs

        # Calculate output values from the inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on params
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

# Dropout class
# This is to prevent our NN from becoming too dependent on any one neuron
class Layer_Dropout:
    def __init__(self, rate):
        # Store rate value
        # Invert this value since if we want dropout of 0.1, we need a success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs, train):
        # Save inputs
        self.inputs = inputs

        # If we are not in training mode, return values
        if not train:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask.
        # Binary mask can be used to isolate and modify particular portions of an image,
        # which can be extremely helpful in many applications, including ours.
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to outputs
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on value
        self.dinputs = dvalues * self.binary_mask

# Input "layer"
class Layer_Input:
    # Forward pass
    def forward(self, inputs, train):
        self.output = inputs


