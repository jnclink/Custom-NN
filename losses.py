# -*- coding: utf-8 -*-

"""
Some loss functions (and their derivatives)
"""

import numpy as np

import utils
from utils import (
    cast,
    check_dtype
)


##############################################################################


# Defining the Categorical Cross-Entropy (CCE) loss function and its derivative


def CCE(y_true, y_pred):
    """
    Categorical Cross-Entropy (CCE) loss function
    
    `y_true` and `y_pred` are 1D vectors or 2D matrices
    
    `y_true` has to be one-hot encoded, and all the values of `y_pred` have to
    be strictly positive (usually those 2 conditions are satisfied when this
    function is called)
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) in [1, 2]
    check_dtype(y_true, utils.DEFAULT_DATATYPE)
    check_dtype(y_pred, utils.DEFAULT_DATATYPE)
    
    if len(y_true.shape) == 1:
        CCE_output = - np.sum(y_true * np.log(y_pred))
    else:
        assert len(y_true.shape) == 2
        
        batch_size = y_true.shape[0]
        CCE_output = np.zeros((batch_size, ), dtype=y_true.dtype)
        
        for batch_sample_index in range(batch_size):
            CCE_output[batch_sample_index] = CCE(y_true[batch_sample_index, :], y_pred[batch_sample_index, :])
    
    check_dtype(CCE_output, utils.DEFAULT_DATATYPE)
    return CCE_output


def CCE_prime(y_true, y_pred):
    """
    Derivative of the Categorical Cross-Entropy (CCE) loss function
    
    `y_true` and `y_pred` are 1D vectors or 2D matrices
    
    `y_true` has to be one-hot encoded, and all the values of `y_pred` have to
    be strictly positive (usually those 2 conditions are satisfied when this
    function is called)
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) in [1, 2]
    check_dtype(y_true, utils.DEFAULT_DATATYPE)
    check_dtype(y_pred, utils.DEFAULT_DATATYPE)
    
    CCE_prime_output = y_pred - y_true
    
    check_dtype(CCE_prime_output, utils.DEFAULT_DATATYPE)
    return CCE_prime_output


##############################################################################


# Defining the Mean Squared Error (MSE) loss function and its derivative


def MSE(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss function
    
    `y_true` and `y_pred` are 1D vectors or 2D matrices
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) in [1, 2]
    check_dtype(y_true, utils.DEFAULT_DATATYPE)
    check_dtype(y_pred, utils.DEFAULT_DATATYPE)
    
    MSE_output = np.mean((y_true - y_pred)**2, axis=-1)
    
    check_dtype(MSE_output, utils.DEFAULT_DATATYPE)
    return MSE_output


def MSE_prime(y_true, y_pred):
    """
    Derivative of the Mean Squared Error (MSE) loss function
    
    `y_true` and `y_pred` are 1D vectors or 2D matrices
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) in [1, 2]
    check_dtype(y_true, utils.DEFAULT_DATATYPE)
    check_dtype(y_pred, utils.DEFAULT_DATATYPE)
    
    scaling_factor = cast(2, utils.DEFAULT_DATATYPE) / cast(y_true.shape[-1], utils.DEFAULT_DATATYPE)
    MSE_prime_output = scaling_factor * (y_pred - y_true)
    
    check_dtype(MSE_prime_output, utils.DEFAULT_DATATYPE)
    return MSE_prime_output

