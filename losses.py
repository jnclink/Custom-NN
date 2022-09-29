# -*- coding: utf-8 -*-

"""
Script defining some loss functions (and their derivatives)
"""

from __future__ import annotations

import numpy as np

import utils
from utils import (
    cast,
    check_dtype,
    _validate_loss_inputs,
    _validate_one_hot_encoded_array
)


##############################################################################


# Defining the Categorical Cross-Entropy (CCE) loss function and its derivative


def CCE(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        y_pred_is_log_softmax_output: bool = False,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Categorical Cross-Entropy (CCE) loss function
    
    `y_true` and `y_pred` are 1D vectors or 2D matrices (usually the latter)
    
    `y_true` has to be one-hot encoded, and all the values of `y_pred` have to
    be in the range ]0, 1] (in practice, those 2 conditions are satisfied when
    this function is called)
    
    By design, `y_pred` is meant to be the output of either the `softmax` or
    the `sigmoid` activation function (from the "activation.py" script)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_loss_inputs(y_true, y_pred)
        assert isinstance(y_pred_is_log_softmax_output, bool)
        
        # `y_true` has to be one-hot encoded
        _validate_one_hot_encoded_array(y_true)
        
        if not(y_pred_is_log_softmax_output):
            # the values of `y_pred` have to be in the range ]0, 1]
            nb_illegal_values_in_y_pred = np.where(y_pred <= 0)[0].size + np.where(y_pred > 1)[0].size
            assert nb_illegal_values_in_y_pred == 0
    
    if not(y_pred_is_log_softmax_output):
        # NB : The `utils.DTYPE_RESOLUTION` correction term is here to replace
        #      the values of `y_pred` that are *very close* to zero with a
        #      small (positive) value
        log_y_pred = np.log(y_pred + utils.DTYPE_RESOLUTION)
    else:
        log_y_pred = y_pred
    
    CCE_output = - np.sum(y_true * log_y_pred, axis=-1)
    
    if enable_checks:
        check_dtype(CCE_output, utils.DEFAULT_DATATYPE)
    return CCE_output


def CCE_prime(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        y_pred_is_log_softmax_output: bool = False,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Derivative of the Categorical Cross-Entropy (CCE) loss function
    
    `y_true` and `y_pred` are 1D vectors or 2D matrices (usually the latter)
    
    `y_true` has to be one-hot encoded, and all the values of `y_pred` have to
    be in the range ]0, 1] (in practice, those 2 conditions are satisfied when
    this function is called)
    
    By design, `y_pred` is meant to be the output of either the `softmax` or
    the `sigmoid` activation function (from the "activation.py" script)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_loss_inputs(y_true, y_pred)
        assert isinstance(y_pred_is_log_softmax_output, bool)
    
    if not(y_pred_is_log_softmax_output):
        used_y_true = y_true
    else:
        # NB : Since `y_true` is one-hot encoded, and since `log(0)` isn't defined,
        #      we have to add a small (positive) correction term to `y_true`
        used_y_true = np.log(y_true + utils.DTYPE_RESOLUTION)
    
    # since the "subtraction" operation doesn't put any constraints on `y_true`
    # and `y_pred`, we will not check if the latter contain any illegal values
    CCE_prime_output = y_pred - used_y_true
    
    if enable_checks:
        check_dtype(CCE_prime_output, utils.DEFAULT_DATATYPE)
    return CCE_prime_output


##############################################################################


# Defining the Mean Squared Error (MSE) loss function and its derivative


def MSE(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        y_pred_is_log_softmax_output: bool = False,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Mean Squared Error (MSE) loss function
    
    `y_true` and `y_pred` are 1D vectors or 2D matrices (usually the latter)
    
    By design, `y_pred` is meant to be the output of either the `softmax` or
    the `sigmoid` activation function (from the "activation.py" script)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_loss_inputs(y_true, y_pred)
        assert isinstance(y_pred_is_log_softmax_output, bool)
    
    if not(y_pred_is_log_softmax_output):
        used_y_true = y_true
    else:
        # NB : Since `y_true` is one-hot encoded, and since `log(0)` isn't defined,
        #      we have to add a small (positive) correction term to `y_true`
        used_y_true = np.log(y_true + utils.DTYPE_RESOLUTION)
    
    MSE_output = np.mean((used_y_true - y_pred)**2, axis=-1)
    
    if enable_checks:
        check_dtype(MSE_output, utils.DEFAULT_DATATYPE)
    return MSE_output


def MSE_prime(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        y_pred_is_log_softmax_output: bool = False,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Derivative of the Mean Squared Error (MSE) loss function
    
    `y_true` and `y_pred` are 1D vectors or 2D matrices (usually the latter)
    
    By design, `y_pred` is meant to be the output of either the `softmax` or
    the `sigmoid` activation function (from the "activation.py" script)
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_loss_inputs(y_true, y_pred)
        assert isinstance(y_pred_is_log_softmax_output, bool)
    
    if not(y_pred_is_log_softmax_output):
        used_y_true = y_true
    else:
        # NB : Since `y_true` is one-hot encoded, and since `log(0)` isn't defined,
        #      we have to add a small (positive) correction term to `y_true`
        used_y_true = np.log(y_true + utils.DTYPE_RESOLUTION)
    
    # depending on the context, `y_true.shape[-1]` is either the number of
    # elements of the inputs (if they are 1D) or the number of classes
    # represented in the input data (if they are 2D - in that case it would
    # imply that `y_true` is one-hot encoded and that all the values of `y_pred`
    # lie in the range [0, 1])
    scaling_factor = cast(2, utils.DEFAULT_DATATYPE) / cast(y_true.shape[-1], utils.DEFAULT_DATATYPE)
    MSE_prime_output = scaling_factor * (y_pred - used_y_true)
    
    if enable_checks:
        check_dtype(MSE_prime_output, utils.DEFAULT_DATATYPE)
    return MSE_prime_output

