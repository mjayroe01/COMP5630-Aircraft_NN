import numpy as np

# Activation ReLU (Rectified Linear Unit)
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs, train):
        # Remember inputs
        self.inputs = inputs
        # Calculate outputs from input values
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Copy values so we can modify original
        self.dinputs = dvalues.copy()

        # Zero gradient where inputs were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

# Softmax Activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs, train):
        # Remember inputs
        self.inputs = inputs

        # Get un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add
            # it to array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
