# -*- coding: utf-8 -*-

"""
Script defining some loss functions (and their derivatives)
"""

import numpy as np

import utils
from utils import (
    cast,
    check_dtype,
    _validate_loss_inputs
)


##############################################################################


# Defining the Categorical Cross-Entropy (CCE) loss function and its derivative


def CCE(y_true, y_pred, check_for_illegal_input_values=True):
    """
    Categorical Cross-Entropy (CCE) loss function
    
    `y_true` and `y_pred` are 1D vectors or 2D matrices
    
    `y_true` has to be one-hot encoded, and all the values of `y_pred` have to
    be in the range ]0, 1] (in practice, those 2 conditions are satisfied when
    this function is called)
    """
    _validate_loss_inputs(y_true, y_pred)
    assert isinstance(check_for_illegal_input_values, bool)
    
    if check_for_illegal_input_values:
        # the values of `y_true` have to be one-hot encoded
        assert y_true.size == np.where(y_true == 0.0)[0].size + np.where(y_true == 1.0)[0].size
        
        # the values of `y_pred` have to be in the range ]0, 1]
        nb_illegal_values_in_y_pred = np.where(y_pred <= 0)[0].size + np.where(y_pred > 1)[0].size
        assert nb_illegal_values_in_y_pred == 0
    
    if len(y_true.shape) == 1:
        CCE_output = - np.sum(y_true * np.log(y_pred))
    elif len(y_true.shape) == 2:
        batch_size = y_true.shape[0]
        CCE_output = np.zeros((batch_size, ), dtype=y_true.dtype)
        
        for batch_sample_index in range(batch_size):
            y_true_sample = y_true[batch_sample_index, :]
            y_pred_sample = y_pred[batch_sample_index, :]
            
            CCE_output[batch_sample_index] = CCE(
                y_true_sample,
                y_pred_sample,
                check_for_illegal_input_values=False
            )
    
    check_dtype(CCE_output, utils.DEFAULT_DATATYPE)
    return CCE_output


def CCE_prime(y_true, y_pred):
    """
    Derivative of the Categorical Cross-Entropy (CCE) loss function
    
    `y_true` and `y_pred` are 1D vectors or 2D matrices
    
    `y_true` has to be one-hot encoded, and all the values of `y_pred` have to
    be in the range ]0, 1] (in practice, those 2 conditions are satisfied when
    this function is called)
    """
    _validate_loss_inputs(y_true, y_pred)
    
    # since the "substraction" operation doesn't put any constraints on `y_true`
    # and `y_pred`, we will not check if the latter contain any illegal values
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
    _validate_loss_inputs(y_true, y_pred)
    
    MSE_output = np.mean((y_true - y_pred)**2, axis=-1)
    
    check_dtype(MSE_output, utils.DEFAULT_DATATYPE)
    return MSE_output


def MSE_prime(y_true, y_pred):
    """
    Derivative of the Mean Squared Error (MSE) loss function
    
    `y_true` and `y_pred` are 1D vectors or 2D matrices
    """
    _validate_loss_inputs(y_true, y_pred)
    
    scaling_factor = cast(2, utils.DEFAULT_DATATYPE) / cast(y_true.shape[-1], utils.DEFAULT_DATATYPE)
    MSE_prime_output = scaling_factor * (y_pred - y_true)
    
    check_dtype(MSE_prime_output, utils.DEFAULT_DATATYPE)
    return MSE_prime_output

