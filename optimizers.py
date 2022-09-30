# -*- coding: utf-8 -*-

"""
Script defining some optimizers that can be used during the training loop
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np

import utils
from utils import (
    cast,
    check_dtype,
    count_nb_decimals_places
)

from regularizers import Regularizer


##############################################################################


class Optimizer(ABC):
    """
    Base (abstract) optimizer class
    """
    def __init__(
            self,
            learning_rate: float,
            *,
            regularizer: Optional[Regularizer] = None
        ) -> None:
        
        # checking the validity of the specified learning rate
        assert isinstance(learning_rate, float)
        learning_rate = round(learning_rate, 6)
        learning_rate = cast(learning_rate, utils.DEFAULT_DATATYPE)
        assert (learning_rate > 0) and (learning_rate < 1)
        
        # generic type representing the global datatype
        Float = np.dtype(utils.DEFAULT_DATATYPE).type
        
        self.learning_rate: Float = learning_rate
        
        # ------------------------------------------------------------------ #
        
        # defining regularizer-related variables
        
        self._has_regularizer: bool = False
        
        # defined for convenience purposes
        self._one = cast(1, utils.DEFAULT_DATATYPE)
        
        self._L1_coeff = None
        self._L2_coeff = None
        
        self._L1_weight_decay_factor = None
        self._L2_weight_decay_factor = None
        
        assert isinstance(regularizer, (type(None), Regularizer))
        self.regularizer = regularizer
        
        if regularizer is not None:
            assert type(regularizer) != Regularizer
            self._has_regularizer = True
            
            if hasattr(regularizer, "L1_coeff"):
                self._L1_coeff = cast(regularizer.L1_coeff, utils.DEFAULT_DATATYPE)
                self._L1_weight_decay_factor = self.learning_rate * self._L1_coeff
                check_dtype(self._L1_weight_decay_factor, utils.DEFAULT_DATATYPE)
            if hasattr(regularizer, "L2_coeff"):
                self._L2_coeff = cast(regularizer.L2_coeff, utils.DEFAULT_DATATYPE)
                self._L2_weight_decay_factor = self._one - self.learning_rate * self._L2_coeff
                check_dtype(self._L2_weight_decay_factor, utils.DEFAULT_DATATYPE)
            
            assert (self._L1_coeff is not None) or (self._L2_coeff is not None)
    
    def __str__(self) -> str:
        # default string representation of the optimizer classes
        precision_learning_rate = max(2, count_nb_decimals_places(self.learning_rate))
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate:.{precision_learning_rate}f})"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __eq__(self, obj: object) -> bool:
        if type(obj) != type(self):
            return False
        return np.allclose(obj.learning_rate, self.learning_rate) and (obj.regularizer == self.regularizer)
    
    def _get_weight_decay_factor(self, weight: np.ndarray) -> Union[float, np.ndarray]:
        """
        Returns the "weight decay factor", i.e. the factor by which the
        weights will be reduced (if there are any L1/L2 regularizers)
        """
        
        # We won't check the input argument, as this method will only be used
        # by the `optimize_weights` method
        
        weight_decay_factor = self._one
        
        if self._L1_coeff is not None:
            weight_decay_factor = self._one - self._L1_weight_decay_factor * np.sign(weight)
        if self._L2_coeff is not None:
            weight_decay_factor *= self._L2_weight_decay_factor
        
        return weight_decay_factor
    
    @abstractmethod
    def optimize_weights(
            self,
            weights: tuple[np.ndarray],
            weight_gradients: tuple[np.ndarray],
            *,
            enable_checks: bool = True
        ) -> list[np.ndarray]:
        """
        Optimizes the given weights using gradient descent (or one of its
        variants), and returns the resulting optimized weights
        """
        pass


##############################################################################


class SgdOptimizer(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer class
    """
    def __init__(
            self,
            learning_rate: float,
            *,
            regularizer: Optional[Regularizer] = None
        ) -> None:
        
        super().__init__(learning_rate, regularizer=regularizer)
    
    def optimize_weights(
            self,
            weights: tuple[np.ndarray],
            weight_gradients: tuple[np.ndarray],
            *,
            enable_checks: bool = True
        ) -> list[np.ndarray]:
        """
        Optimizes the given weights using the "basic"/stochastic gradient
        descent algorihtm, and returns the resulting optimized weights
        """
        # ------------------------------------------------------------------ #
        
        # checking the specified arguments
        
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            assert isinstance(weights, (list, tuple))
            assert len(weights) > 0
            assert isinstance(weight_gradients, (list, tuple))
            assert len(weight_gradients) == len(weights)
        
        # ------------------------------------------------------------------ #
        
        optimized_weights = []
        
        for weight, weight_gradient in zip(weights, weight_gradients):
            if enable_checks:
                if np.isscalar(weight):
                    assert np.isscalar(weight_gradient)
                else:
                    assert isinstance(weight, np.ndarray) and isinstance(weight_gradient, np.ndarray)
                    assert len(weight.shape) == 2
                    assert weight.shape == weight_gradient.shape
                check_dtype(weight, utils.DEFAULT_DATATYPE)
                check_dtype(weight_gradient, utils.DEFAULT_DATATYPE)
            
            # -------------------------------------------------------------- #
            
            # GRADIENT DESCENT
            
            if not(self._has_regularizer):
                decayed_weight = weight
            else:
                weight_decay_factor = self._get_weight_decay_factor(weight)
                decayed_weight = weight_decay_factor * weight
            
            optimized_weight = decayed_weight - self.learning_rate * weight_gradient
            
            # -------------------------------------------------------------- #
            
            if enable_checks:
                check_dtype(optimized_weight, utils.DEFAULT_DATATYPE)
            
            optimized_weights.append(optimized_weight)
        
        return optimized_weights


##############################################################################


class AdamOptimizer(Optimizer):
    """
    Adam optimizer class
    """
    def __init__(
            self,
            learning_rate: float,
            *,
            regularizer: Optional[Regularizer] = None
        ) -> None:
        
        super().__init__(learning_rate, regularizer=regularizer)
        
        # generic type representing the global datatype
        Float = np.dtype(utils.DEFAULT_DATATYPE).type
        
        # first order momentum (default value)
        self.beta_1: Float = cast(0.9, utils.DEFAULT_DATATYPE)
        assert (self.beta_1 > 0) and (self.beta_1 < 1)
        
        # second order momentum (default value)
        self.beta_2: Float = cast(0.999, utils.DEFAULT_DATATYPE)
        assert (self.beta_2 > 0) and (self.beta_2 < 1)
        
        self._beta_1_inverse: Float = self._one - self.beta_1
        self._beta_2_inverse: Float = self._one - self.beta_2
        check_dtype(self._beta_1_inverse, utils.DEFAULT_DATATYPE)
        check_dtype(self._beta_2_inverse, utils.DEFAULT_DATATYPE)
        
        # by default (used for numerical stability)
        self.epsilon: Float = cast(1e-5, utils.DEFAULT_DATATYPE)
        assert (self.epsilon > 0) and (self.epsilon < 1e-2)
        
        # if the scaling factors become *really close* to one, then there's
        # no point in computing them anymore
        self._stop_computing_scaling_factor_1: bool = False
        self._stop_computing_scaling_factor_2: bool = False
        
        # initializing the list that will contain the first and second moments
        self.first_moments:  list[Union[Float, np.ndarray]] = []
        self.second_moments: list[Union[Float, np.ndarray]] = []
        
        # initializing the timestep
        self.t: int = 0
    
    def optimize_weights(
            self,
            weights: tuple[np.ndarray],
            weight_gradients: tuple[np.ndarray],
            *,
            enable_checks: bool = True
        ) -> list[np.ndarray]:
        """
        Optimizes the given weights using the "Adam" gradient descent algorihtm,
        and returns the resulting optimized weights
        """
        # ------------------------------------------------------------------ #
        
        # checking the specified arguments
        
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            assert isinstance(weights, (list, tuple))
            assert len(weights) > 0
            assert isinstance(weight_gradients, (list, tuple))
            assert len(weight_gradients) == len(weights)
        
        # ------------------------------------------------------------------ #
        
        optimized_weights = []
        
        # updating the timestep
        self.t += 1
        
        # for convenience purposes, we compute those two scaling factors ahead
        # of time
        if not(self._stop_computing_scaling_factor_1):
            scaling_factor_1 = self._one / (self._one - cast(self.beta_1**self.t, utils.DEFAULT_DATATYPE))
            self._stop_computing_scaling_factor_1 = np.allclose(scaling_factor_1, 1)
        else:
            scaling_factor_1 = self._one
        if not(self._stop_computing_scaling_factor_2):
            scaling_factor_2 = self._one / (self._one - cast(self.beta_2**self.t, utils.DEFAULT_DATATYPE))
            self._stop_computing_scaling_factor_2 = np.allclose(scaling_factor_2, 1)
        else:
            scaling_factor_2 = self._one
        
        if enable_checks:
            check_dtype(scaling_factor_1, utils.DEFAULT_DATATYPE)
            check_dtype(scaling_factor_2, utils.DEFAULT_DATATYPE)
        
        for weight_index, (weight, weight_gradient) in enumerate(zip(weights, weight_gradients)):
            if enable_checks:
                if np.isscalar(weight):
                    assert np.isscalar(weight_gradient)
                else:
                    assert isinstance(weight, np.ndarray) and isinstance(weight_gradient, np.ndarray)
                    assert len(weight.shape) == 2
                    assert weight.shape == weight_gradient.shape
                check_dtype(weight, utils.DEFAULT_DATATYPE)
                check_dtype(weight_gradient, utils.DEFAULT_DATATYPE)
            
            # -------------------------------------------------------------- #
            
            # initializing the first and second moment vectors/scalars for the
            # current weight (at the very first timestep)
            
            if self.t == 1:
                if np.isscalar(weight):
                    zero = cast(0, utils.DEFAULT_DATATYPE)
                    self.first_moments.append(zero)
                    self.second_moments.append(zero)
                else:
                    output_size = weight.shape[1]
                    zeros = np.zeros((1, output_size), dtype=utils.DEFAULT_DATATYPE)
                    self.first_moments.append(zeros)
                    self.second_moments.append(zeros.copy())
            
            # -------------------------------------------------------------- #
            
            # GRADIENT DESCENT
            
            # computing the (bias-corrected) first moment estimate
            first_moment = self.first_moments[weight_index]
            first_moment_estimate = self.beta_1 * first_moment + self._beta_1_inverse * weight_gradient
            self.first_moments[weight_index] = first_moment_estimate
            bias_corrected_first_moment_estimate = scaling_factor_1 * first_moment_estimate
            
            # computing the (bias-corrected) second moment estimate
            second_moment = self.second_moments[weight_index]
            second_moment_estimate = self.beta_2 * second_moment + self._beta_2_inverse * np.square(weight_gradient)
            self.second_moments[weight_index] = second_moment_estimate
            bias_corrected_second_moment_estimate = scaling_factor_2 * second_moment_estimate
            
            optimized_weight_gradient = bias_corrected_first_moment_estimate / (np.sqrt(bias_corrected_second_moment_estimate) + self.epsilon)
            
            if not(self._has_regularizer):
                decayed_weight = weight
            else:
                weight_decay_factor = self._get_weight_decay_factor(weight)
                decayed_weight = weight_decay_factor * weight
            
            optimized_weight = decayed_weight - self.learning_rate * optimized_weight_gradient
            
            # -------------------------------------------------------------- #
            
            if enable_checks:
                check_dtype(optimized_weight, utils.DEFAULT_DATATYPE)
            
            optimized_weights.append(optimized_weight)
        
        return optimized_weights


##############################################################################


class RMSpropOptimizer(Optimizer):
    """
    RMSprop optimizer class
    """
    def __init__(
            self,
            learning_rate: float,
            *,
            regularizer: Optional[Regularizer] = None
        ) -> None:
        
        super().__init__(learning_rate, regularizer=regularizer)
        
        # generic type representing the global datatype
        Float = np.dtype(utils.DEFAULT_DATATYPE).type
        
        # momentum (default value)
        self.beta: Float = cast(0.9, utils.DEFAULT_DATATYPE)
        assert (self.beta > 0) and (self.beta < 1)
        
        self._beta_inverse: Float = self._one - self.beta
        check_dtype(self._beta_inverse, utils.DEFAULT_DATATYPE)
        
        # by default (used for numerical stability)
        self.epsilon: Float = cast(1e-5, utils.DEFAULT_DATATYPE)
        assert (self.epsilon > 0) and (self.epsilon < 1e-2)
        
        # initializing the list that will contain the moments
        self.moments: list[Union[Float, np.ndarray]] = []
        
        self._moments_are_initialized: bool = False
    
    def optimize_weights(
            self,
            weights: tuple[np.ndarray],
            weight_gradients: tuple[np.ndarray],
            *,
            enable_checks: bool = True
        ) -> list[np.ndarray]:
        """
        Optimizes the given weights using the "RMSprop" gradient descent algorihtm,
        and returns the resulting optimized weights
        """
        # ------------------------------------------------------------------ #
        
        # checking the specified arguments
        
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            assert isinstance(weights, (list, tuple))
            assert len(weights) > 0
            assert isinstance(weight_gradients, (list, tuple))
            assert len(weight_gradients) == len(weights)
        
        # ------------------------------------------------------------------ #
        
        optimized_weights = []
        
        for weight_index, (weight, weight_gradient) in enumerate(zip(weights, weight_gradients)):
            if enable_checks:
                if np.isscalar(weight):
                    assert np.isscalar(weight_gradient)
                else:
                    assert isinstance(weight, np.ndarray) and isinstance(weight_gradient, np.ndarray)
                    assert len(weight.shape) == 2
                    assert weight.shape == weight_gradient.shape
                check_dtype(weight, utils.DEFAULT_DATATYPE)
                check_dtype(weight_gradient, utils.DEFAULT_DATATYPE)
            
            # -------------------------------------------------------------- #
            
            # initializing the moment vectors/scalars for the current weight
            # (at the very first iteration)
            
            if not(self._moments_are_initialized):
                if np.isscalar(weight):
                    zero = cast(0, utils.DEFAULT_DATATYPE)
                    self.moments.append(zero)
                else:
                    output_size = weight.shape[1]
                    zeros = np.zeros((1, output_size), dtype=utils.DEFAULT_DATATYPE)
                    self.moments.append(zeros)
            
            # -------------------------------------------------------------- #
            
            # GRADIENT DESCENT
            
            # computing the moment estimate
            moment = self.moments[weight_index]
            moment_estimate = self.beta * moment + self._beta_inverse * np.square(weight_gradient)
            self.moments[weight_index] = moment_estimate
            
            optimized_weight_gradient = weight_gradient / (np.sqrt(moment_estimate) + self.epsilon)
            
            if not(self._has_regularizer):
                decayed_weight = weight
            else:
                weight_decay_factor = self._get_weight_decay_factor(weight)
                decayed_weight = weight_decay_factor * weight
            
            optimized_weight = decayed_weight - self.learning_rate * optimized_weight_gradient
            
            # -------------------------------------------------------------- #
            
            if enable_checks:
                check_dtype(optimized_weight, utils.DEFAULT_DATATYPE)
            
            optimized_weights.append(optimized_weight)
        
        if not(self._moments_are_initialized):
            self._moments_are_initialized = True
        
        return optimized_weights

