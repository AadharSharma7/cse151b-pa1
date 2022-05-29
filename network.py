from termios import IEXTEN
from tkinter import Y
import numpy as np
import data
import time
import tqdm
import sys
import math


"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""


def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """
    # clip any extreme values so we don't get NaN from calculations
    a = np.clip(a, -709, 709)
    
    return 1/(1+np.exp(-a))


def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / sum (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """
    # clip any extreme values so we don't get NaN from calculations
    a = np.clip(a, -709, 709)
    
    return np.exp(a)/np.sum(np.exp(a))


def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.

    # L(x) = t*ln(y) + (1-t)*ln(1-y)
    L(x) = -summ (t*ln(y) + (1-t)*ln(1-y))
    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float
        binary cross entropy loss value according to above definition
    """
    # t=0 and y=1 --> target is 0, prediction is 1 (100% prob for 1)
    # t=1 and y=0 --> target is 1, prediction is 0 --> (0% prob for 1, 100% prob for 0)
    # add a small constant to "stabilize" the log input (don't want to do log0)
    a = t*np.log(y+0.00001) + (1-t)*np.log((1-y)+0.00001)
    out = -np.sum(a)

    return out


def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    # L(x) = - sum sum (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float
        multiclass cross entropy loss value according to above definition
    """
    return -np.sum(t*np.log(y+0.00001))

class Network:
    def __init__(self, hyperparameters, activation, loss, out_dim):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        """
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss

        self.weights = np.zeros((28*28+1, out_dim))

    def get_weights(self):
        return self.weights

    def forward(self, X):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = sig(wT*x)
            where
                sig = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """
        z = X@self.weights
        return self.activation(z)

    def __call__(self, X):
        return self.forward(X)

    # Function to convert predicted outputs and labels to 0/1
    def binary_classification_helper(self, prediction_probs, decimal_ys):
        """
        Parameters
        ----------
            prediction_probs : the model's predicted outputs (probabilities)
            decimal_ys : the true labels as class numbers
        
        Returns
        -------
            predictions : the model's predicted outputs, converted to 0 or 1
            targets : the true labels, converted to 0 or 1
        """

        # if predicted probability is >= 0.5, convert that to 1; otherwise convert to 0
        p_index = prediction_probs >= 0.5
        predictions = p_index.astype(float)

        # convert class 6 labels to be 1
        t_index = decimal_ys == 6
        targets = t_index.astype(float)

        return predictions, targets

    # Function to train the network
    def train(self, minibatch, binary=False):
        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        tuple containing:
            average loss over minibatch
            accuracy over minibatch
        """
        
        # get input and true labels from current minibatch
        X, y = minibatch

        batch_size = self.hyperparameters.batch_size
        lr = self.hyperparameters.learning_rate

        # append 1's to the input so our biases work
        X = data.append_bias(X) 
        
        # Used later to get the target values: 0 or 1 depending on if
        # the decimal value is 0/2 or 6
        # if decimal value is 0/2 then target is 0, else target is 1
        decimal_ys = data.onehot_decode(y)

        # get model's predictions (as probabilities)
        prediction_probs = self.forward(X)

        # Getting the target values
        if binary:
            # say batch size = 4
            # X shape = 4x785
            # y shape = 4x7
            # prediction_probs shape = 4x1
            prediction_probs = prediction_probs.flatten()
            predictions, targets = self.binary_classification_helper(
                prediction_probs, decimal_ys)  # Get the predictions and targets as 1s or 0s

            gradient = -((targets - prediction_probs) @ X)  # Gradient is size 785
            gradient = np.array([gradient]) # Make gradient size 1x785
            # weights is 785x1, so need to transpose gradient
            # update weights
            self.weights = self.weights - (lr*gradient).T
        
        # multiclass training
        else:
            # say batch size = 4
            # X shape = 4x785
            # y shape = 4x10
            # prediction_probs shape = 4x10
            pred_diffs = y - prediction_probs # 4x10

            gradient = -(pred_diffs.T @ X) # 10x785

            # now update weights
            # 785x10 - 785x10
            self.weights = self.weights - (lr*gradient).T

        # _loss = self.loss(prediction_probs, targets)

        # return _loss, _acc

    def test(self, minibatch, binary=False):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """
        X, y = minibatch

        # batch_size = self.hyperparameters.batch_size
        # lr = self.hyperparameters.learning_rate

        X = data.append_bias(X)
        decimal_ys = data.onehot_decode(y)

        prediction_probs = self.forward(X)

        if binary:
            prediction_probs = prediction_probs.flatten()

            predictions, targets = self.binary_classification_helper(
                prediction_probs, decimal_ys)

            difference = targets - predictions
        
        # multiclass testing
        else:
            decoded_y = data.onehot_decode(y)
            predictions = data.onehot_decode(prediction_probs)

            difference = decoded_y - predictions

            targets = y

        correct = (difference == 0).sum()
        total = len(difference)
        _acc = float(correct)/float(total)
        _loss = self.loss(prediction_probs, targets) / len(X) # divide by x to normalize against test size

        return _loss, _acc
