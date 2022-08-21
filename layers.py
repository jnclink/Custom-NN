# -*- coding: utf-8 -*-

"""
Main layer-type classes
"""

import sys
import numpy as np
from abc import ABC, abstractmethod

from activations import (
    ReLU, ReLU_prime,
    leaky_ReLU, leaky_ReLU_prime,
    tanh, tanh_prime,
    softmax, softmax_prime,
    sigmoid, sigmoid_prime
)


##############################################################################


class Layer(ABC):
    """
    Base (abstract) layer class
    """
    def __init__(self):
        self.input  = None
        self.output = None
        self.nb_trainable_params = None
    
    @abstractmethod
    def forward_propagation(self, input_data, training=True):
        """
        Computes the output Y of a layer for a given input X
        """
        pass
    
    @abstractmethod
    def backward_propagation(self, output_error, learning_rate):
        """
        Computes dE/dX for a given dE/dY, and updates the trainable parameters
        if there are any
        """
        pass


##############################################################################


class InputLayer(Layer):
    """
    Input layer
    """
    def __init__(self, input_size):
        assert input_size > 0
        self.input_size = input_size
        self.nb_trainable_params = 0
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.input_size})"
    
    def forward_propagation(self, input_data, training=True):
        self.input  = input_data
        self.output = self.input
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        """
        NB : Here, `learning_rate` is not used because there are no trainable
             parameters
        """
        input_error = output_error
        return input_error


##############################################################################


class DenseLayer(Layer):
    """
    Dense (i.e. fully connected) layer
    """
    def __init__(self, nb_neurons, seed=None):
        assert nb_neurons > 0
        self.output_size = nb_neurons
        self.seed = seed
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.output_size})"
    
    def build(self, input_size):
        """
        Now that we know the input size of the layer, we can actually initialize
        the latter. This layer is built when it is added to the network (with
        the `Network.add` method of the `network.py` script)
        """
        assert input_size > 0
        self.input_size = input_size
        
        # using the "He initialization" (because it works well with the ReLU activation)
        np.random.seed(self.seed)
        std = np.sqrt(2.0 / self.input_size)
        self.weights = std * np.random.randn(self.input_size, self.output_size)
        self.biases  = std * np.random.randn(1, self.output_size)
        np.random.seed(None) # resetting the seed
        
        # = self.input_size * self.output_size + self.output_size
        # = (self.input_size + 1) * self.output_size
        self.nb_trainable_params = self.weights.size + self.biases.size
    
    def forward_propagation(self, input_data, training=True):
        self.input  = input_data
        
        # duplicating the biases for batch processing
        batch_size = input_data.shape[0]
        duplicated_biases = np.tile(self.biases, (batch_size, 1))
        
        self.output = self.input @ self.weights + duplicated_biases
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        """
        Computes dE/dW and dE/dB for a given output_error=dE/dY, and returns
        input_error=dE/dX
        """
        input_error = output_error @ self.weights.T # = dE/dX
        
        # error averaging for batch processing
        batch_size = output_error.shape[0]
        weights_error = (1.0 / batch_size) * self.input.T @ output_error # = dE/dW
        biases_error = np.mean(output_error, axis=0, keepdims=True) # = dE/dB
        
        # updating the trainable parameters
        self.weights -= learning_rate * weights_error
        self.biases  -= learning_rate * biases_error
        
        return input_error


##############################################################################


class ActivationLayer(Layer):
    """
    Activation layer
    """
    
    # class variable
    AVAILABLE_ACTIVATIONS = {
        "relu"       : (ReLU, ReLU_prime),
        "leaky_relu" : (leaky_ReLU, leaky_ReLU_prime),
        "tanh"       : (tanh, tanh_prime),
        "softmax"    : (softmax, softmax_prime),
        "sigmoid"    : (sigmoid, sigmoid_prime)
    }
    
    def __init__(self, activation_name, **kwargs):
        assert isinstance(activation_name, str)
        activation_name = activation_name.lower()
        if activation_name not in list(self.AVAILABLE_ACTIVATIONS.keys()):
            print(f"\nActivationLayer.__init__ - ERROR - Unrecognized activation name : \"{activation_name}\"")
            sys.exit(-1)
        
        self.activation_name = activation_name
        self.activation, self.activation_prime = self.AVAILABLE_ACTIVATIONS[self.activation_name]
        
        if self.activation == leaky_ReLU:
            default_leaky_ReLU_coeff = 0.01
            self.activation_kwargs = {
                "leaky_ReLU_coeff" : kwargs.get("leaky_ReLU_coeff", default_leaky_ReLU_coeff)
            }
        else:
            self.activation_kwargs = {}
        
        # NB : Since the softmax activation only applies to VECTORS (and not
        #      scalars), the backpropagation formula won't be the same as the other
        #      activations. Essentially, the element-wise multiplication becomes
        #      an actual matrix multiplication (cf. the `backward_propagation` method)
        self.is_softmax = (self.activation == softmax)
        
        self.nb_trainable_params = 0
    
    def __str__(self):
        return f"{self.__class__.__name__}(\"{self.activation_name}\")"
    
    def forward_propagation(self, input_data, training=True):
        """
        Returns the activated input
        """
        self.input  = input_data
        self.output = self.activation(self.input, **self.activation_kwargs)
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        """
        NB : Here, `learning_rate` is not used because there are no trainable
             parameters
        """
        activation_prime_of_input = self.activation_prime(self.input, **self.activation_kwargs)
        
        if not(self.is_softmax):
            # element-wise multiplication
            input_error = output_error * activation_prime_of_input
        else:
            batch_size = output_error.shape[0]
            
            input_error = np.zeros(output_error.shape, dtype=output_error.dtype)
            for batch_sample_index in range(batch_size):
                # matrix multiplication (NOT element-wise multiplication)
                input_error[batch_sample_index, :] = output_error[batch_sample_index, :] @ activation_prime_of_input[batch_sample_index]
            
            """
            The previous code block is completely equivalent to the following line :
            
            input_error = (output_error.reshape(batch_size, 1, output_error.shape[1]) @ activation_prime_of_input).reshape(output_error.shape)
            
            --> Basically, we're using 3D matrix multiplication tricks to make
                the computations a bit faster !
            """
        
        return input_error


##############################################################################


class DropoutLayer(Layer):
    """
    Dropout regularization layer
    """
    def __init__(self, dropout_rate, seed=None):
        self.dropout_rate = dropout_rate
        assert (self.dropout_rate > 0) and (self.dropout_rate < 1)
        
        self.seed = seed
        
        # by default
        self.deactivated_value = 0.0
        
        # all the non-deactivated values will be scaled up by this factor (by default)
        self.scaling_factor = 1.0 / (1 - self.dropout_rate)
        
        self.nb_trainable_params = 0
    
    def __str__(self):
        precision = 2 # by default
        return f"{self.__class__.__name__}({self.dropout_rate:.{precision}f})"
    
    def generate_random_dropout_matrix(self, dims, dtype):
        # `dims` refers to the dimensions of the (2D) dropout matrix
        assert len(dims) == 2
        
        dropout_matrix = self.scaling_factor * np.ones(dims, dtype=dtype)
        nb_of_values_to_drop_per_input_sample = max(int(self.dropout_rate * dims[1]), 1)
        
        choices_for_dropped_indices = np.arange(dims[1])
        
        np.random.seed(self.seed)
        for iteration_index in range(dims[0]):
            indices_of_randomly_dropped_values = np.random.choice(choices_for_dropped_indices, size=(nb_of_values_to_drop_per_input_sample, ))
            dropout_matrix[iteration_index, indices_of_randomly_dropped_values] = self.deactivated_value
        np.random.seed(None) # resetting the seed
        
        return dropout_matrix
    
    def forward_propagation(self, input_data, training=True):
        """
        Returns the input with randomly deactivated neurons/nodes
        """
        self.input = input_data
        
        if training:
            # NB : The dropout matrix is randomly re-generated from scratch
            #      EVERY TIME the forwarding method is called (during the
            #      training phase)
            self.dropout_matrix  = self.generate_random_dropout_matrix(
                self.input.shape,
                self.input.dtype
            )
            self.output = self.input * self.dropout_matrix
        else:
            # the "Dropout process" doesn't apply to the validation and testing sets
            self.output = self.input
        
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        """
        NB : Here, `learning_rate` is not used because there are no trainable
             parameters
        """
        input_error = output_error * self.dropout_matrix
        return input_error

