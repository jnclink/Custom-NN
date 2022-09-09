# -*- coding: utf-8 -*-

"""
Script defining the activation functions (and their derivatives)
"""

import numpy as np

import utils
from utils import (
    cast,
    check_dtype,
    _validate_activation_input,
    _validate_leaky_ReLU_coeff
)


##############################################################################


# Defining the Rectified Linear Unit (ReLU) activation function and its derivative


def ReLU(x):
    """
    Rectified Linear Unit (ReLU) activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    """
    _validate_activation_input(x)
    
    ReLU_output = np.maximum(x, 0)
    
    check_dtype(ReLU_output, utils.DEFAULT_DATATYPE)
    return ReLU_output


def ReLU_prime(x):
    """
    Derivative of the Rectified Linear Unit (ReLU) activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    """
    _validate_activation_input(x)
    
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
    _validate_activation_input(x)
    leaky_ReLU_coeff = _validate_leaky_ReLU_coeff(leaky_ReLU_coeff)
    
    leaky_ReLU_output = np.maximum(x, leaky_ReLU_coeff * x)
    
    check_dtype(leaky_ReLU_output, utils.DEFAULT_DATATYPE)
    return leaky_ReLU_output


def leaky_ReLU_prime(x, leaky_ReLU_coeff=0.01):
    """
    Derivative of the leaky Rectified Linear Unit (leaky ReLU) activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    Usually, `leaky_ReLU_coeff` is a small positive constant
    """
    _validate_activation_input(x)
    leaky_ReLU_coeff = _validate_leaky_ReLU_coeff(leaky_ReLU_coeff)
    
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
    _validate_activation_input(x)
    
    tanh_output = np.tanh(x)
    
    check_dtype(tanh_output, utils.DEFAULT_DATATYPE)
    return tanh_output


def tanh_prime(x):
    """
    Derivative of the hyperbolic tangent (tanh) activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    """
    _validate_activation_input(x)
    
    one = cast(1, utils.DEFAULT_DATATYPE)
    tanh_prime_output = one - np.tanh(x)**2
    
    check_dtype(tanh_prime_output, utils.DEFAULT_DATATYPE)
    return tanh_prime_output


##############################################################################


# Defining the softmax activation function and its derivative


def softmax(x, check_for_illegal_output_values=True):
    """
    Softmax activation function
    
    `x` is a 1D vector or a 2D matrix (usually the latter)
    By definition, here `x` CANNOT be a scalar
    """
    _validate_activation_input(x, can_be_scalar=False)
    assert isinstance(check_for_illegal_output_values, bool)
    
    if len(x.shape) == 1:
        # NB : The softmax function is translationally invariant, i.e., for any
        #      scalar `t`, we will have : `softmax(x) = softmax(x - t)`.
        #      Therefore, in order to prevent potential overflow errors, we can
        #      simply set `t = max(x)`. Indeed, in that case, `x - t` will have
        #      negative values, and therefore `exp(x - t)` (and `sum(exp(x - t))`)
        #      will never raise overflow errors !
        exps = np.exp(x - np.max(x))
        softmax_output = exps / np.sum(exps)
    elif len(x.shape) == 2:
        batch_size = x.shape[0]
        softmax_output = np.zeros(x.shape, dtype=x.dtype)
        for batch_sample_index in range(batch_size):
            x_sample = x[batch_sample_index, :]
            softmax_output[batch_sample_index, :] = softmax(
                x_sample,
                check_for_illegal_output_values=False
            )
    
    if check_for_illegal_output_values:
        # replacing the softmax output values that are *very close* to zero
        # with the smallest possible positive value of the current global datatype
        resolution = utils.DTYPE_RESOLUTION
        softmax_output[softmax_output < resolution] = resolution
        
        # the values of the softmax output have to be strictly positive
        nb_illegal_values_in_softmax_output = np.where(softmax_output <= 0)[0].size
        assert nb_illegal_values_in_softmax_output == 0
    
    check_dtype(softmax_output, utils.DEFAULT_DATATYPE)
    return softmax_output


def softmax_prime(x):
    """
    Derivative of the softmax activation function
    
    `x` is a 1D vector or a 2D matrix (usually the latter)
    By definition, here `x` CANNOT be a scalar
    """
    _validate_activation_input(x, can_be_scalar=False)
    
    if len(x.shape) == 1:
        softmax_output = softmax(x).reshape((1, x.shape[0]))
        softmax_prime_output = np.diagflat(softmax_output) - softmax_output.T @ softmax_output
        return softmax_prime_output
    elif len(x.shape) == 2:
        batch_size = x.shape[0]
        softmax_prime_output = np.zeros((batch_size, x.shape[1], x.shape[1]), dtype=x.dtype)
        for batch_sample_index in range(batch_size):
            x_sample = x[batch_sample_index, :]
            softmax_prime_output[batch_sample_index] = softmax_prime(x_sample)
    
    check_dtype(softmax_prime_output, utils.DEFAULT_DATATYPE)
    return softmax_prime_output


##############################################################################


# Defining the sigmoid activation function and its derivative


def sigmoid(x, check_for_illegal_output_values=True):
    """
    Sigmoid activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    """
    _validate_activation_input(x)
    assert isinstance(check_for_illegal_output_values, bool)
    
    if np.isscalar(x) or (len(x.shape) == 1):
        one = cast(1, utils.DEFAULT_DATATYPE)
        sigmoid_output = one / (one + np.exp(-x))
    elif len(x.shape) == 2:
        sigmoid_output = np.zeros(x.shape, dtype=x.dtype)
        batch_size = x.shape[0]
        for batch_sample_index in range(batch_size):
            x_sample = x[batch_sample_index, :]
            sigmoid_output[batch_sample_index, :] = sigmoid(
                x_sample,
                check_for_illegal_output_values=False
            )
    
    if check_for_illegal_output_values:
        # replacing the sigmoid output values that are *very close* to zero
        # with the smallest possible positive value of the current global datatype
        resolution = utils.DTYPE_RESOLUTION
        sigmoid_output[sigmoid_output < resolution] = resolution
        
        # the values of the sigmoid output have to be strictly positive
        nb_illegal_values_in_sigmoid_output = np.where(sigmoid_output <= 0)[0].size
        assert nb_illegal_values_in_sigmoid_output == 0
    
    check_dtype(sigmoid_output, utils.DEFAULT_DATATYPE)
    return sigmoid_output


def sigmoid_prime(x):
    """
    Derivative of the sigmoid activation function
    
    `x` is either a scalar, a 1D vector or a 2D matrix (usually the latter)
    """
    _validate_activation_input(x)
    
    if np.isscalar(x) or (len(x.shape) == 1):
        sigmoid_output = sigmoid(x)
        one = cast(1, utils.DEFAULT_DATATYPE)
        sigmoid_prime_output = sigmoid_output * (one - sigmoid_output)
    elif len(x.shape) == 2:
        sigmoid_prime_output = np.zeros(x.shape, dtype=x.dtype)
        batch_size = x.shape[0]
        for batch_sample_index in range(batch_size):
            x_sample = x[batch_sample_index, :]
            sigmoid_prime_output[batch_sample_index, :] = sigmoid_prime(x_sample)
    
    check_dtype(sigmoid_prime_output, utils.DEFAULT_DATATYPE)
    return sigmoid_prime_output

