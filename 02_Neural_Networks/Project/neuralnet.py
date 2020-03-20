"""
Bike Sharing Prediction - Udacity Deep Learning Nanodegree Project

The general skeleton of this class was provided. See the following
repository for the original files and more details:

    https://github.com/udacity/deep-learning-v2-pytorch
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(
            0.0, self.input_nodes ** -0.5, (self.input_nodes, self.hidden_nodes)
        )

        self.weights_hidden_to_output = np.random.normal(
            0.0, self.hidden_nodes ** -0.5, (self.hidden_nodes, self.output_nodes)
        )

        self.lr = learning_rate

        # Sigmoid activation function
        self.activation_function = lambda x: 1.0 / (1.0 + np.exp(-x))

    def train(self, features, targets):
        """
        Train the network on batch of features and targets.

        Parameters
        ----------
        features:
            2D array, each row is one data record, each column is a feature
        targets:
            1D array of target values
        """

        n_records = features.shape[0]

        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        # Loop over data (SGD)
        for X, y in zip(features, targets):

            # Forward pass
            final_outputs, hidden_outputs = self.forward(X)

            # Backpropagation
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(
                final_outputs,
                hidden_outputs,
                X,
                y,
                delta_weights_i_h,
                delta_weights_h_o,
            )

        # Weight update
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward(self, X):
        """
        Forward pass

        Parameters
        ---------
        X:
            batch of features
        """

        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(
        self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o
    ):
        """
        Backpropagation

        Parameters
        ----------
        final_outputs: 
            output from forward pass
        y: 
            target (i.e. label) batch
        delta_weights_i_h:
            change in weights from input to hidden layers
        delta_weights_h_o:
            change in weights from hidden to output layers

        """

        # Error (loss)
        error = y - final_outputs
        output_error_term = error  # No sigmoid on output

        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        hidden_error_term = (
            hidden_error * hidden_outputs * (1 - hidden_outputs)
        )  # Sigmoid

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]

        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """
        Update weights on gradient descent step

        Parameters
        ---------
        delta_weights_i_h:
            change in weights from input to hidden layers
        delta_weights_h_o:
            change in weights from hidden to output layers
        n_records:
            number of records
        """
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records

        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        """
        Run forward pass

        Parameters
        ----------
        features:
            1D array of feature values
        """

        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs
