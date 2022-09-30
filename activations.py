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
    
    ReLU_prime_output = np.ones(x.shape, dtype=x.dtype)
    ReLU_prime_output[x <= 0] = 0
    
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
        used_x = np.expand_dims(x, 0)
    elif len(x.shape) == 2:
        used_x = x
    
    batch_size, output_size = used_x.shape
    softmax_prime_output = np.zeros((batch_size, output_size, output_size), dtype=x.dtype)
    
    if input_is_activation_output:
        softmax_output = np.expand_dims(used_x, 1)
    else:
        softmax_output = np.expand_dims(softmax(used_x, enable_checks=False), 1)
    
    I = np.tile(np.identity(output_size, dtype=utils.DEFAULT_DATATYPE), (batch_size, 1, 1))
    softmax_prime_output = (I - softmax_output) * np.swapaxes(softmax_output, 1, 2)
    
    if len(x.shape) == 1:
        softmax_prime_output = np.squeeze(softmax_prime_output)
    
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
        used_x = np.expand_dims(x, 0)
    elif len(x.shape) == 2:
        used_x = x
    
    batch_size, output_size = used_x.shape
    log_softmax_prime_output = np.zeros((batch_size, output_size, output_size), dtype=x.dtype)
    
    # NB : To compute the derivative of the log-softmax, we only need the
    #      output of the softmax function (and not the output of the
    #      log-softmax function)
    if input_is_activation_output:
        # exp(log_softmax) = exp(log(softmax)) = softmax
        softmax_output = np.expand_dims(np.exp(used_x), 1)
    else:
        softmax_output = np.expand_dims(softmax(used_x, enable_checks=False), 1)
    
    I = np.tile(np.identity(output_size, dtype=utils.DEFAULT_DATATYPE), (batch_size, 1, 1))
    log_softmax_prime_output = I - softmax_output
    
    if len(x.shape) == 1:
        log_softmax_prime_output = np.squeeze(log_softmax_prime_output)
    
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

