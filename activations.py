# -*- coding: utf-8 -*-

"""
Script defining some activation functions (and their derivatives)
"""

from __future__ import annotations

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


def ReLU(
        x: np.ndarray,
        *,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
    
    ReLU_output = np.maximum(x, 0)
    
    if enable_checks:
        check_dtype(ReLU_output, utils.DEFAULT_DATATYPE)
    return ReLU_output


def ReLU_prime(
        x: np.ndarray,
        *,
        input_is_activation_output: bool = False, # not used here
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Derivative of the Rectified Linear Unit (ReLU) activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
    
    ReLU_prime_output = np.zeros(x.shape, dtype=x.dtype)
    ReLU_prime_output[x > 0] = 1
    
    if enable_checks:
        check_dtype(ReLU_prime_output, utils.DEFAULT_DATATYPE)
    return ReLU_prime_output


##############################################################################


# Defining the leaky Rectified Linear Unit (leaky ReLU) activation function
# and its derivative


def leaky_ReLU(
        x: np.ndarray,
        *,
        leaky_ReLU_coeff: float = 0.01,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Leaky Rectified Linear Unit (leaky ReLU) activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter),
    and `leaky_ReLU_coeff` is a small positive constant
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
        leaky_ReLU_coeff = _validate_leaky_ReLU_coeff(leaky_ReLU_coeff)
    elif isinstance(leaky_ReLU_coeff, float):
        leaky_ReLU_coeff = cast(leaky_ReLU_coeff, utils.DEFAULT_DATATYPE)
    
    leaky_ReLU_output = np.maximum(x, 0) + leaky_ReLU_coeff * np.minimum(x, 0)
    
    if enable_checks:
        check_dtype(leaky_ReLU_output, utils.DEFAULT_DATATYPE)
    return leaky_ReLU_output


def leaky_ReLU_prime(
        x: np.ndarray,
        *,
        leaky_ReLU_coeff: float = 0.01,
        input_is_activation_output: bool = False, # not used here
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Derivative of the leaky Rectified Linear Unit (leaky ReLU) activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter),
    and `leaky_ReLU_coeff` is a small positive constant
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
        leaky_ReLU_coeff = _validate_leaky_ReLU_coeff(leaky_ReLU_coeff)
    elif isinstance(leaky_ReLU_coeff, float):
        leaky_ReLU_coeff = cast(leaky_ReLU_coeff, utils.DEFAULT_DATATYPE)
    
    leaky_ReLU_prime_output = np.ones(x.shape, dtype=x.dtype)
    leaky_ReLU_prime_output[x <= 0] = leaky_ReLU_coeff
    
    if enable_checks:
        check_dtype(leaky_ReLU_prime_output, utils.DEFAULT_DATATYPE)
    return leaky_ReLU_prime_output


##############################################################################


# Defining the hyperbolic tangent (tanh) activation function and its derivative


def tanh(
        x: np.ndarray,
        *,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Hyperbolic tangent (tanh) activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
    
    tanh_output = np.tanh(x)
    
    if enable_checks:
        check_dtype(tanh_output, utils.DEFAULT_DATATYPE)
    return tanh_output


def tanh_prime(
        x: np.ndarray,
        *,
        input_is_activation_output: bool = False,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Derivative of the hyperbolic tangent (tanh) activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
        assert isinstance(input_is_activation_output, bool)
    
    one = cast(1, utils.DEFAULT_DATATYPE)
    
    if input_is_activation_output:
        tanh_prime_output = one - x**2
    else:
        tanh_prime_output = one - tanh(x, enable_checks=False)**2
    
    if enable_checks:
        check_dtype(tanh_prime_output, utils.DEFAULT_DATATYPE)
    return tanh_prime_output


##############################################################################


# Defining the softmax activation function and its derivative


def softmax(
        x: np.ndarray,
        *,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Softmax activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
    
    # NB : The softmax function is translationally invariant, i.e., for any
    #      scalar `t`, we will have : `softmax(x) = softmax(x - t)`.
    #      Therefore, in order to prevent potential overflow errors, we can
    #      simply set `t = max(x)`. Indeed, in that case, `x - t` will have
    #      negative values, and therefore `exp(x - t)` (and `sum(exp(x - t))`)
    #      will never raise overflow errors !
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    softmax_output = exps / np.sum(exps, axis=-1, keepdims=True)
    
    if enable_checks:
        # the values of the softmax output have to be strictly positive
        nb_illegal_values_in_softmax_output = np.where(softmax_output <= 0)[0].size
        assert nb_illegal_values_in_softmax_output == 0
        
        check_dtype(softmax_output, utils.DEFAULT_DATATYPE)
    
    return softmax_output


def softmax_prime(
        x: np.ndarray,
        *,
        input_is_activation_output: bool = False,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Derivative of the softmax activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    
    NB : If `x` is 1D, the output will be 2D, and if `x` is 2D, then the
         output will be 3D
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
        assert isinstance(input_is_activation_output, bool)
    
    if len(x.shape) == 1:
        if input_is_activation_output:
            softmax_output = x.reshape((1, x.size))
        else:
            softmax_output = softmax(x, enable_checks=False).reshape((1, x.size))
        
        softmax_prime_output = np.diag(softmax_output[0]) - softmax_output.T @ softmax_output
    
    elif len(x.shape) == 2:
        batch_size, output_size = x.shape
        softmax_prime_output = np.zeros((batch_size, output_size, output_size), dtype=x.dtype)
        
        for batch_sample_index in range(batch_size):
            x_sample = x[batch_sample_index, :]
            
            # recursive call
            softmax_prime_output[batch_sample_index] = softmax_prime(
                x_sample,
                input_is_activation_output=input_is_activation_output,
                enable_checks=False
            )
    
    if enable_checks:
        check_dtype(softmax_prime_output, utils.DEFAULT_DATATYPE)
    return softmax_prime_output


##############################################################################


# Defining the logarithmic softmax activation function and its derivative


def log_softmax(
        x: np.ndarray,
        *,
        use_approximation: bool = False,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Logarithmic softmax activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
        assert isinstance(use_approximation, bool)
    
    approximation_of_log_softmax_output = x - np.max(x, axis=-1, keepdims=True)
    
    if use_approximation:
        log_softmax_output = approximation_of_log_softmax_output
    else:
        correction_term = np.log(np.sum(np.exp(approximation_of_log_softmax_output), axis=-1, keepdims=True))
        log_softmax_output = approximation_of_log_softmax_output - correction_term
    
    if enable_checks:
        check_dtype(log_softmax_output, utils.DEFAULT_DATATYPE)
    return log_softmax_output


def log_softmax_prime(
        x: np.ndarray,
        *,
        input_is_activation_output: bool = False,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Derivative of the logarithmic softmax activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    
    NB : If `x` is 1D, the output will be 2D, and if `x` is 2D, then the
         output will be 3D
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
        assert isinstance(input_is_activation_output, bool)
    
    if len(x.shape) == 1:
        I = np.identity(x.size)
        
        if input_is_activation_output:
            log_softmax_prime_output = I - np.exp(x)
        else:
            log_softmax_prime_output = I - softmax(x, enable_checks=False)
    
    elif len(x.shape) == 2:
        batch_size, output_size = x.shape
        log_softmax_prime_output = np.zeros((batch_size, output_size, output_size), dtype=x.dtype)
        
        for batch_sample_index in range(batch_size):
            x_sample = x[batch_sample_index, :]
            
            # recursive call
            log_softmax_prime_output[batch_sample_index] = log_softmax_prime(
                x_sample,
                input_is_activation_output=input_is_activation_output,
                enable_checks=False
            )
    
    if enable_checks:
        check_dtype(log_softmax_prime_output, utils.DEFAULT_DATATYPE)
    return log_softmax_prime_output


##############################################################################


# Defining the sigmoid activation function and its derivative


def sigmoid(
        x: np.ndarray,
        *,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Sigmoid activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
    
    one = cast(1, utils.DEFAULT_DATATYPE)
    sigmoid_output = one / (one + np.exp(-x))
    
    if enable_checks:
        # the values of the sigmoid output have to be strictly positive
        nb_illegal_values_in_sigmoid_output = np.where(sigmoid_output <= 0)[0].size
        assert nb_illegal_values_in_sigmoid_output == 0
        
        check_dtype(sigmoid_output, utils.DEFAULT_DATATYPE)
    
    return sigmoid_output


def sigmoid_prime(
        x: np.ndarray,
        *,
        input_is_activation_output: bool = False,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Derivative of the sigmoid activation function
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_activation_input(x)
        assert isinstance(input_is_activation_output, bool)
    
    if input_is_activation_output:
        sigmoid_output = x
    else:
        sigmoid_output = sigmoid(x, enable_checks=False)
    
    one = cast(1, utils.DEFAULT_DATATYPE)
    sigmoid_prime_output = sigmoid_output * (one - sigmoid_output)
    
    if enable_checks:
        check_dtype(sigmoid_prime_output, utils.DEFAULT_DATATYPE)
    return sigmoid_prime_output

