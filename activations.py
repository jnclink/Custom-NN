# -*- coding: utf-8 -*-

"""
Some activation functions and their derivatives
"""

import numpy as np


##############################################################################

# NB : Here, `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)

##############################################################################


def ReLU(x):
    return np.maximum(x, 0)


def ReLU_prime(x):
    if np.isscalar(x):
        if x >= 0:
            return 1
        return 0
    
    # here, `x` is a vector/matrix
    ReLU_prime_x = x.copy()
    ReLU_prime_x[x >= 0] = 1
    ReLU_prime_x[x < 0] = 0
    return ReLU_prime_x


##############################################################################


def leaky_ReLU(x, leaky_ReLU_coeff=0.01):
    return np.maximum(x, leaky_ReLU_coeff * x)


def leaky_ReLU_prime(x, leaky_ReLU_coeff=0.01):
    if np.isscalar(x):
        if x >= 0:
            return 1
        return leaky_ReLU_coeff
    
    # here, `x` is a vector/matrix
    leaky_ReLU_prime_x = x.copy()
    leaky_ReLU_prime_x[x >= 0] = 1
    leaky_ReLU_prime_x[x < 0] = leaky_ReLU_coeff
    return leaky_ReLU_prime_x


##############################################################################


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x)**2


##############################################################################


# NB : For the softmax activation, `x` CANNOT be a scalar (by definition)


def softmax(x):
    if len(x.shape) == 1:
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
    
    assert len(x.shape) == 2
    batch_size = x.shape[0]
    softmax_x = np.zeros(x.shape, dtype=x.dtype)
    for batch_sample_index in range(batch_size):
        softmax_x[batch_sample_index, :] = softmax(x[batch_sample_index, :])
    return softmax_x


def softmax_prime(x):
    if len(x.shape) == 1:
        softmax_x = softmax(x).reshape((1, x.shape[0]))
        return np.diagflat(softmax_x) - softmax_x.T @ softmax_x
    
    assert len(x.shape) == 2
    batch_size  = x.shape[0]
    softmax_prime_x = np.zeros((batch_size, x.shape[1], x.shape[1]), dtype=x.dtype)
    for batch_sample_index in range(batch_size):
        softmax_prime_x[batch_sample_index] = softmax_prime(x[batch_sample_index, :])
    return softmax_prime_x


##############################################################################


def sigmoid(x):
    if np.isscalar(x) or (len(x.shape) == 1):
        return 1 / (1 + np.exp(-x))
    
    # here, `x` is a 2D matrix
    assert len(x.shape) == 2
    sigmoid_x = np.zeros(x.shape, dtype=x.dtype)
    batch_size = x.shape[0]
    for batch_sample_index in range(batch_size):
        sigmoid_x[batch_sample_index, :] = sigmoid(x[batch_sample_index, :])
    return sigmoid_x


def sigmoid_prime(x):
    if np.isscalar(x) or (len(x.shape) == 1):
        sigmoid_x = sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    # here, `x` is a 2D matrix
    assert len(x.shape) == 2
    sigmoid_prime_x = np.zeros(x.shape, dtype=x.dtype)
    batch_size = x.shape[0]
    for batch_sample_index in range(batch_size):
        sigmoid_prime_x[batch_sample_index, :] = sigmoid_prime(x[batch_sample_index, :])
    return sigmoid_prime_x

