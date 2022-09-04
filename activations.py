# -*- coding: utf-8 -*-

"""
Some activation functions (and their derivatives)
"""

import numpy as np

import utils
from utils import (
    cast,
    check_dtype
)


##############################################################################


# Defining the Rectified Linear Unit (ReLU) activation function and its derivative


def ReLU(x):
    """
    Rectified Linear Unit (ReLU) activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    """
    check_dtype(x, utils.DEFAULT_DATATYPE)
    
    ReLU_output = np.maximum(x, 0)
    
    check_dtype(ReLU_output, utils.DEFAULT_DATATYPE)
    return ReLU_output


def ReLU_prime(x):
    """
    Derivative of the Rectified Linear Unit (ReLU) activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    """
    check_dtype(x, utils.DEFAULT_DATATYPE)
    
    if np.isscalar(x):
        if x >= 0:
            return cast(1, utils.DEFAULT_DATATYPE)
        return cast(0, utils.DEFAULT_DATATYPE)
    
    # here, `x` is a vector/matrix
    
    ReLU_prime_output = np.zeros(x.shape, dtype=x.dtype)
    ReLU_prime_output[x >= 0] = 1
    
    check_dtype(ReLU_prime_output, utils.DEFAULT_DATATYPE)
    return ReLU_prime_output


##############################################################################


# Defining the leaky Rectified Linear Unit (leaky ReLU) activation function
# and its derivative


def leaky_ReLU(x, leaky_ReLU_coeff=0.01):
    """
    Leaky Rectified Linear Unit (leaky ReLU) activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    Usually, `leaky_ReLU_coeff` is a small positive constant
    """
    check_dtype(x, utils.DEFAULT_DATATYPE)
    leaky_ReLU_coeff = cast(leaky_ReLU_coeff, utils.DEFAULT_DATATYPE)
    
    leaky_ReLU_output = np.maximum(x, leaky_ReLU_coeff * x)
    
    check_dtype(leaky_ReLU_output, utils.DEFAULT_DATATYPE)
    return leaky_ReLU_output


def leaky_ReLU_prime(x, leaky_ReLU_coeff=0.01):
    """
    Derivative of the leaky Rectified Linear Unit (leaky ReLU) activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    Usually, `leaky_ReLU_coeff` is a small positive constant
    """
    check_dtype(x, utils.DEFAULT_DATATYPE)
    leaky_ReLU_coeff = cast(leaky_ReLU_coeff, utils.DEFAULT_DATATYPE)
    
    if np.isscalar(x):
        if x >= 0:
            return cast(1, utils.DEFAULT_DATATYPE)
        
        # NB : `leaky_ReLU_coeff` has already been cast to `utils.DEFAULT_DATATYPE`
        return leaky_ReLU_coeff
    
    # here, `x` is a vector/matrix
    
    leaky_ReLU_prime_output = np.ones(x.shape, dtype=x.dtype)
    leaky_ReLU_prime_output[x < 0] = leaky_ReLU_coeff
    
    check_dtype(leaky_ReLU_prime_output, utils.DEFAULT_DATATYPE)
    return leaky_ReLU_prime_output


##############################################################################


# Defining the hyperbolic tangent (tanh) activation function and its derivative


def tanh(x):
    """
    Hyperbolic tangent (tanh) activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    """
    check_dtype(x, utils.DEFAULT_DATATYPE)
    
    tanh_output = np.tanh(x)
    
    check_dtype(tanh_output, utils.DEFAULT_DATATYPE)
    return tanh_output


def tanh_prime(x):
    """
    Derivative of the hyperbolic tangent (tanh) activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    """
    check_dtype(x, utils.DEFAULT_DATATYPE)
    
    one = cast(1, utils.DEFAULT_DATATYPE)
    tanh_prime_output = one - np.tanh(x)**2
    
    check_dtype(tanh_prime_output, utils.DEFAULT_DATATYPE)
    return tanh_prime_output


##############################################################################


# Defining the softmax activation function and its derivative


def softmax(x):
    """
    Softmax activation function
    
    `x` is a 1D vector or a 2D matrix (usually the latter)
    By definition, here `x` CANNOT be a scalar
    """
    assert isinstance(x, np.ndarray)
    check_dtype(x, utils.DEFAULT_DATATYPE)
    
    if len(x.shape) == 1:
        # NB : The softmax function is translationally invariant, i.e., for any
        #      scalar `t`, we will have : `softmax(x) = softmax(x - t)`.
        #      Therefore, in order to prevent potential overflow errors, we can
        #      simply set `t = max(x)`. Indeed, in that case, `x - t` will have
        #      negative values, and therefore `exp(x - t)` (and `sum(exp(x - t))`)
        #      will never raise overflow errors !
        exps = np.exp(x - np.max(x))
        softmax_output = exps / np.sum(exps)
    else:
        assert len(x.shape) == 2
        batch_size = x.shape[0]
        softmax_output = np.zeros(x.shape, dtype=x.dtype)
        for batch_sample_index in range(batch_size):
            softmax_output[batch_sample_index, :] = softmax(x[batch_sample_index, :])
        
        # replacing the softmax output values that are *very close* to zero
        # with the smallest possible positive value of the current global datatype
        resolution = utils.DTYPE_RESOLUTION
        softmax_output[softmax_output < resolution] = resolution
        
        nb_negative_values_in_softmax_output = np.where(softmax_output <= 0)[0].size
        assert nb_negative_values_in_softmax_output == 0
    
    check_dtype(softmax_output, utils.DEFAULT_DATATYPE)
    return softmax_output


def softmax_prime(x):
    """
    Derivative of the softmax activation function
    
    `x` is a 1D vector or a 2D matrix (usually the latter)
    By definition, here `x` CANNOT be a scalar
    """
    assert isinstance(x, np.ndarray)
    check_dtype(x, utils.DEFAULT_DATATYPE)
    
    if len(x.shape) == 1:
        softmax_output = softmax(x).reshape((1, x.shape[0]))
        softmax_prime_output = np.diagflat(softmax_output) - softmax_output.T @ softmax_output
        return softmax_prime_output
    else:
        assert len(x.shape) == 2
        batch_size  = x.shape[0]
        softmax_prime_output = np.zeros((batch_size, x.shape[1], x.shape[1]), dtype=x.dtype)
        for batch_sample_index in range(batch_size):
            softmax_prime_output[batch_sample_index] = softmax_prime(x[batch_sample_index, :])
    
    check_dtype(softmax_prime_output, utils.DEFAULT_DATATYPE)
    return softmax_prime_output


##############################################################################


# Defining the sigmoid activation function and its derivative


def sigmoid(x):
    """
    Sigmoid activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    """
    check_dtype(x, utils.DEFAULT_DATATYPE)
    
    if np.isscalar(x) or (len(x.shape) == 1):
        one = cast(1, utils.DEFAULT_DATATYPE)
        sigmoid_output = one / (one + np.exp(-x))
    else:
        # here, `x` is a 2D matrix
        assert len(x.shape) == 2
        sigmoid_output = np.zeros(x.shape, dtype=x.dtype)
        batch_size = x.shape[0]
        for batch_sample_index in range(batch_size):
            sigmoid_output[batch_sample_index, :] = sigmoid(x[batch_sample_index, :])
        
        # replacing the sigmoid output values that are *very close* to zero
        # with the smallest possible positive value of the current global datatype
        resolution = utils.DTYPE_RESOLUTION
        sigmoid_output[sigmoid_output < resolution] = resolution
        
        nb_negative_values_in_sigmoid_output = np.where(sigmoid_output <= 0)[0].size
        assert nb_negative_values_in_sigmoid_output == 0
    
    check_dtype(sigmoid_output, utils.DEFAULT_DATATYPE)
    return sigmoid_output


def sigmoid_prime(x):
    """
    Derivative of the sigmoid activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    """
    check_dtype(x, utils.DEFAULT_DATATYPE)
    
    if np.isscalar(x) or (len(x.shape) == 1):
        sigmoid_output = sigmoid(x)
        one = cast(1, utils.DEFAULT_DATATYPE)
        sigmoid_prime_output = sigmoid_output * (one - sigmoid_output)
    else:
        # here, `x` is a 2D matrix
        assert len(x.shape) == 2
        sigmoid_prime_output = np.zeros(x.shape, dtype=x.dtype)
        batch_size = x.shape[0]
        for batch_sample_index in range(batch_size):
            sigmoid_prime_output[batch_sample_index, :] = sigmoid_prime(x[batch_sample_index, :])
    
    check_dtype(sigmoid_prime_output, utils.DEFAULT_DATATYPE)
    return sigmoid_prime_output

