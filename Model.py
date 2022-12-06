import copy
import pickle
import numpy as np
from Layers import Layer_Input
from Activation_Functions import Activation_Softmax
from Metric_Functions import Loss_CategoricalCrossEntropy, Activation_Softmax_Loss_CategoricalCrossEntropy


class Model:
    def __init__(self):
        # Create list of network objects
        self.layers = []
        # Softmax classifier's output obj
        self.softmax_classifier_output = None

    # Add objects to model
    def add(self, layer):
        self.layers.append(layer)

    # Set loss, optimizer, and accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Finalize model
    def finalize(self):
        # Create and set input layer
        self.input_layer = Layer_Input()

        # Count all objects
        layer_count = len(self.layers)

        # Initialize a list of trainable layers
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):
            # If first layer, previous layer is the input
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # All layers except first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # Last layer - next object is loss
            # Will also save reference to the last object
            # whose output is the output of the model
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains "weights" attribute,
            # is it a trainable layer.
            # Add to list of trainable layers
            # No need to check for biases, since checking weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            # Update loss object w/ trainable layers
            if self.loss is not None:
                self.loss.remember_trainable_layers(self.trainable_layers)

            # If output activation is Softmax and loss is Categorical Cross Entropy
            # create an object with combined activation and loss functions
            # with faster calculation of gradient
            if isinstance(self.layers[-1], Activation_Softmax) and \
                    isinstance(self.loss, Loss_CategoricalCrossEntropy):
                self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()

    # Training time!
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # initialize accuracy
        self.accuracy.init(y)

        # Default value if batch size is not set
        train_steps = 1

        # If validation data is passed, set default steps for validation also
        if validation_data is not None:
            validation_steps = 1

            # readability
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # The above is floor division, so it will divide down
            # If there's remaining data but not a full batch, it will not be included
            # Add '1' to include this partial batch
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size

                # The above is floor division, so it will divide down
                # If there's remaining data but not a full batch, it will not be included
                # Add '1' to include this partial batch
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # main training loop
        for epoch in range(1, epochs + 1):
            # Print out epoch number
            print(f'epoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):
                # If batch size is not set
                # train using one set and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # otherwise slice a batch
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                # Do forward pass
                output = self.forward(batch_X, train=True)

                # Calculate loss
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate accuracy (temp removed this to save time)
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Do backward pass
                self.backward(output, batch_y)

                # Optimize
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(
                    include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'accuracy: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data loss: {epoch_data_loss:.3f}, ' +
                  f'regularization loss: {epoch_regularization_loss:.3f}), ' +
                  f'learning rate: {self.optimizer.current_learning_rate}')

            # If validation data
            if validation_data is not None:
                # Evaluate model
                self.evaluate(*validation_data, batch_size=batch_size)

    # Forward pass
    def forward(self, X, train):
        # Call forward method on input layer.
        # Will set output property that first layer in previous object is expecting
        self.input_layer.forward(X, train)

        # Call forward method of every object in chain
        # Pass output of previous object as param
        for layer in self.layers:
            layer.forward(layer.prev.output, train)

        # "layer" is now last object from list, return output
        return layer.output

    # Backward pass
    def backward(self, output, y):
        # If softmax
        if self.softmax_classifier_output is not None:
            # First, call backward on combined activation/loss.
            # This will set dinputs value.
            self.softmax_classifier_output.backward(output, y)

            # We will not call backward on last layer (which is softmax)
            # since we used the combined activation/loss object.
            # Set dinputs value in this object
            self.layers[- 1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through all objects
            # except the last one in reverse order, using dinputs as a param
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        # Call backward on loss, which will set dinputs property.
        # The last layer will attempt to access this shortly.
        self.loss.cbackward(output, y)

        # Call backward method going through all objects in reverse order
        # using dinputs as a param
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Evaluation method
    def evaluate(self, X_val, y_val, *, batch_size=None):
        # Default value if batch size not set
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Above is floor division. If there's remaining data, but not a full
            # batch, then it won't be included.
            # Add '1' to include this partial batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated loss and accuracy values
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for step in range(validation_steps):
            # If batch size not set, train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            # Otherwise slice a batch
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            # Do forward pass
            output = self.forward(batch_X, train=False)

            # Calculate loss
            self.loss.calculate(output, batch_y)

            # Get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print summary
        print(f'validation, ' +
              f'accuracy: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    # Prediction method
    # This method will take a trained model and a
    # never-before-seen image and try to predict
    # the correct classification.
    def predict(self, X, *, batch_size=None):
        # Default if batch_size is not set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            # The above is floor division, so it will divide down
            # If there's remaining data but not a full batch, it will not be included
            # Add '1' to include this partial batch
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        # Model outputs
        output = []

        # Iterate over steps
        for step in range(prediction_steps):
            # If batch_size not set,
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X

            # Otherwise slice a batch
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]

            # Do forward pass
            batch_ouput = self.forward(batch_X, train=False)

            # Attach batch prediction to list of predictions
            output.append(batch_ouput)

        # Stack and return output
        return np.vstack(output)

    # Method to save our model
    def save(self, path):
        # Make deep copy of current model
        model = copy.deepcopy(self)

        # Reset accumulated values for loss and accuracy
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data in input layer
        # Reset gradients, if any exist
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For every layer, remove inputs, output, and dinputs
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open file in binary write mode and save model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        # Open file in binary read mode and load model
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model
