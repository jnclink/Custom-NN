# -*- coding: utf-8 -*-

"""
Script defining some optimizers that can be used during the training loop
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

import utils
from utils import (
    cast,
    check_dtype,
    count_nb_decimals_places
)


##############################################################################


class Optimizer(ABC):
    """
    Base (abstract) optimizer class
    """
    def __init__(self, learning_rate: float) -> None:
        # checking the validity of the specified learning rate
        assert isinstance(learning_rate, float)
        learning_rate = round(learning_rate, 6)
        learning_rate = cast(learning_rate, utils.DEFAULT_DATATYPE)
        assert (learning_rate > 0) and (learning_rate < 1)
        
        # generic type representing the global datatype
        Float = np.dtype(utils.DEFAULT_DATATYPE).type
        
        self.learning_rate: Float = learning_rate
    
    def __str__(self) -> str:
        # default string representation of the optimizer classes
        precision_learning_rate = max(2, count_nb_decimals_places(self.learning_rate))
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate:.{precision_learning_rate}f})"
    
    def __repr__(self) -> str:
        return str(self)
    
    @abstractmethod
    def optimize_weights(
            self,
            weights: Union[list[Union[np.float32, np.float64, np.ndarray]], tuple[Union[np.float32, np.float64, np.ndarray]]],
            weight_gradients: Union[list[Union[np.float32, np.float64, np.ndarray]], tuple[Union[np.float32, np.float64, np.ndarray]]],
            *,
            enable_checks: bool = True
        ) -> list[Union[np.float32, np.float64, np.ndarray]]:
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
    def __init__(self, learning_rate: float) -> None:
        super().__init__(learning_rate)
    
    def optimize_weights(
            self,
            weights: Union[list[Union[np.float32, np.float64, np.ndarray]], tuple[Union[np.float32, np.float64, np.ndarray]]],
            weight_gradients: Union[list[Union[np.float32, np.float64, np.ndarray]], tuple[Union[np.float32, np.float64, np.ndarray]]],
            *,
            enable_checks: bool = True
        ) -> list[Union[np.float32, np.float64, np.ndarray]]:
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
            
            # gradient descent
            
            optimized_weight = weight - self.learning_rate * weight_gradient
            
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
    def __init__(self, learning_rate: float) -> None:
        super().__init__(learning_rate)
        
        # generic type representing the global datatype
        Float = np.dtype(utils.DEFAULT_DATATYPE).type
        
        # first order momentum (default value)
        self.beta_1: Float = cast(0.9, utils.DEFAULT_DATATYPE)
        assert (self.beta_1 > 0) and (self.beta_1 < 1)
        
        # second order momentum (default value)
        self.beta_2: Float = cast(0.999, utils.DEFAULT_DATATYPE)
        assert (self.beta_2 > 0) and (self.beta_2 < 1)
        
        one: Float = cast(1, utils.DEFAULT_DATATYPE)
        self._inverse_beta_1: Float = one - self.beta_1
        self._inverse_beta_2: Float = one - self.beta_2
        check_dtype(self._inverse_beta_1, utils.DEFAULT_DATATYPE)
        check_dtype(self._inverse_beta_2, utils.DEFAULT_DATATYPE)
        
        # by default (used for numerical stability)
        self.epsilon: Float = cast(1e-5, utils.DEFAULT_DATATYPE)
        assert (self.epsilon > 0) and (self.epsilon < 1e-2)
        
        # initializing the list that will contain the first and second moments
        self.first_moments:  list[Union[Float, np.ndarray]] = []
        self.second_moments: list[Union[Float, np.ndarray]] = []
        
        # initializing the timestep
        self.t: int = 0
    
    def optimize_weights(
            self,
            weights: Union[list[Union[np.float32, np.float64, np.ndarray]], tuple[Union[np.float32, np.float64, np.ndarray]]],
            weight_gradients: Union[list[Union[np.float32, np.float64, np.ndarray]], tuple[Union[np.float32, np.float64, np.ndarray]]],
            *,
            enable_checks: bool = True
        ) -> list[Union[np.float32, np.float64, np.ndarray]]:
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
        one = cast(1, utils.DEFAULT_DATATYPE)
        scaling_factor_1 = one / (one - cast(self.beta_1**self.t, utils.DEFAULT_DATATYPE))
        if np.allclose(scaling_factor_1, 1):
            scaling_factor_1 = one
        scaling_factor_2 = one / (one - cast(self.beta_2**self.t, utils.DEFAULT_DATATYPE))
        if np.allclose(scaling_factor_2, 1):
            scaling_factor_2 = one
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
            
            # gradient descent
            
            # computing the (bias-corrected) first moment estimate
            first_moment = self.first_moments[weight_index]
            first_moment_estimate = self.beta_1 * first_moment + self._inverse_beta_1 * weight_gradient
            self.first_moments[weight_index] = first_moment_estimate
            bias_corrected_first_moment_estimate = scaling_factor_1 * first_moment_estimate
            
            # if `weight_gradient` is a float32, then its square (using the
            # `**` operator) will be a float64, which explains why we don't
            # use the `**` operator here (the same goes for the `np.power` method)
            square_of_weight_gradient = weight_gradient * weight_gradient
            
            # computing the (bias-corrected) second moment estimate
            second_moment = self.second_moments[weight_index]
            second_moment_estimate = self.beta_2 * second_moment + self._inverse_beta_2 * square_of_weight_gradient
            self.second_moments[weight_index] = second_moment_estimate
            bias_corrected_second_moment_estimate = scaling_factor_2 * second_moment_estimate
            
            optimized_weight_gradient = bias_corrected_first_moment_estimate / (np.sqrt(bias_corrected_second_moment_estimate) + self.epsilon)
            
            optimized_weight = weight - self.learning_rate * optimized_weight_gradient
            
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
    def __init__(self, learning_rate: float) -> None:
        super().__init__(learning_rate)
        
        # generic type representing the global datatype
        Float = np.dtype(utils.DEFAULT_DATATYPE).type
        
        # momentum (default value)
        self.beta: Float = cast(0.9, utils.DEFAULT_DATATYPE)
        assert (self.beta > 0) and (self.beta < 1)
        
        one: Float = cast(1, utils.DEFAULT_DATATYPE)
        self._inverse_beta: Float = one - self.beta
        check_dtype(self._inverse_beta, utils.DEFAULT_DATATYPE)
        
        # by default (used for numerical stability)
        self.epsilon: Float = cast(1e-5, utils.DEFAULT_DATATYPE)
        assert (self.epsilon > 0) and (self.epsilon < 1e-2)
        
        # initializing the list that will contain the moments
        self.moments: list[Union[Float, np.ndarray]] = []
        
        self._moments_are_initialized: bool = False
    
    def optimize_weights(
            self,
            weights: Union[list[Union[np.float32, np.float64, np.ndarray]], tuple[Union[np.float32, np.float64, np.ndarray]]],
            weight_gradients: Union[list[Union[np.float32, np.float64, np.ndarray]], tuple[Union[np.float32, np.float64, np.ndarray]]],
            *,
            enable_checks: bool = True
        ) -> list[Union[np.float32, np.float64, np.ndarray]]:
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
            
            # gradient descent
            
            # if `weight_gradient` is a float32, then its square (using the
            # `**` operator) will be a float64, which explains why we don't
            # use the `**` operator here (the same goes for the `np.power` method)
            square_of_weight_gradient = weight_gradient * weight_gradient
            
            # computing the moment estimate
            moment = self.moments[weight_index]
            moment_estimate = self.beta * moment + self._inverse_beta * square_of_weight_gradient
            self.moments[weight_index] = moment_estimate
            
            optimized_weight_gradient = weight_gradient / (np.sqrt(moment_estimate) + self.epsilon)
            
            optimized_weight = weight - self.learning_rate * optimized_weight_gradient
            
            # -------------------------------------------------------------- #
            
            if enable_checks:
                check_dtype(optimized_weight, utils.DEFAULT_DATATYPE)
            
            optimized_weights.append(optimized_weight)
        
        if not(self._moments_are_initialized):
            self._moments_are_initialized = True
        
        return optimized_weights

