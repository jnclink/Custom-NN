# -*- coding: utf-8 -*-

"""
Some loss functions and their derivatives
"""

import numpy as np


##############################################################################

# NB : Here, `y_true` and `y_pred` are 1D vectors or 2D matrices

##############################################################################


# MSE = Mean Square Error


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2, axis=-1)


def MSE_prime(y_true, y_pred):
    nb_classes = y_true.shape[-1]
    return (2 / nb_classes) * (y_pred - y_true)


##############################################################################


# CCE = Categorical Cross-Entropy

# NB : For the CCE, `y_true` has to be one-hot encoded, and all the values of
#      `y_pred` have to be strictly positive (usually those 2 conditions are
#      satisfied when the CCE is called)


def CCE(y_true, y_pred):
    if len(y_true.shape) == 1:
        assert len(y_pred.shape) == 1
        return - np.sum(y_true * np.log(y_pred))
    
    assert len(y_true.shape) == 2
    assert len(y_pred.shape) == 2
    
    batch_size = y_true.shape[0]
    CCE_vector = np.zeros((batch_size, ), dtype=y_true.dtype)
    
    for batch_sample_index in range(batch_size):
        CCE_vector[batch_sample_index] = CCE(y_true[batch_sample_index, :], y_pred[batch_sample_index, :])
    
    return CCE_vector


def CCE_prime(y_true, y_pred):
    return y_pred - y_true

