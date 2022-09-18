# -*- coding: utf-8 -*-

"""
Script defining the main layer classes
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Callable

import numpy as np

import utils
from utils import (
    cast,
    check_dtype,
    _validate_numpy_dtype,
    _validate_leaky_ReLU_coeff,
    list_to_string,
    count_nb_decimals_places
)

from activations import (
    ReLU,       ReLU_prime,
    leaky_ReLU, leaky_ReLU_prime,
    tanh,       tanh_prime,
    softmax,    softmax_prime,
    sigmoid,    sigmoid_prime
)


##############################################################################


class Layer(ABC):
    """
    Base (abstract) layer class
    """
    
    def __init__(self) -> None:
        self.input:  Union[None, np.ndarray] = None
        self.output: Union[None, np.ndarray] = None
        self.nb_trainable_params: Union[None, int] = None
    
    def __str__(self) -> str:
        # default string representation of the layer classes (most of the
        # time their `__str__` method will override this one)
        return f"{self.__class__.__name__}()"
    
    def __repr__(self) -> str:
        return str(self)
    
    @abstractmethod
    def forward_propagation(
            self,
            input_data: np.ndarray,
            training: bool = True,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Computes the output Y of a layer for a given input X. The `training`
        kwarg indicates whether we're currently in the training phase or not
        (`training` will only be set to `False` during the validation and testing
         phases)
        """
        pass
    
    def _validate_forward_propagation_inputs(
            self,
            input_data: np.ndarray,
            training: bool
        ) -> None:
        """
        Checks if `input_data` and `training` are valid or not
        """
        assert isinstance(input_data, np.ndarray)
        assert len(input_data.shape) == 2
        check_dtype(input_data, utils.DEFAULT_DATATYPE)
        
        assert isinstance(training, bool)
    
    def build(self, input_size: int) -> None:
        """
        Now that we know the input size of the layer, we can actually
        initialize/build the latter (if needed). The layer will be built when
        it is added to the network (with the `Network.add` method of the
        "network.py" script)
        
        This method doesn't have to be overridden by any subclasses,
        therefore it won't be defined as an abstract method
        """
        pass
    
    @abstractmethod
    def backward_propagation(
            self,
            output_gradient: np.ndarray,
            learning_rate: float,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Computes dE/dX for a given dE/dY, and updates the trainable parameters
        if there are any (using gradient descent)
        """
        pass
    
    def _validate_backward_propagation_inputs(
            self,
            output_gradient: np.ndarray,
            learning_rate: float
        ) -> None:
        """
        Checks if `output_gradient` and `learning_rate` are valid or not
        """
        assert isinstance(output_gradient, np.ndarray)
        assert len(output_gradient.shape) == 2
        check_dtype(output_gradient, utils.DEFAULT_DATATYPE)
        
        assert isinstance(learning_rate, float)
        learning_rate = cast(learning_rate, utils.DEFAULT_DATATYPE)
        
        learning_rate = max(learning_rate, utils.DTYPE_RESOLUTION)
        assert (learning_rate > 0) and (learning_rate < 1)
        
        return learning_rate


##############################################################################


class InputLayer(Layer):
    """
    Input layer class. The only purpose of this class is to be the very first
    layer of the network, so that it can signal to the next layer what the
    input size of the network is
    """
    def __init__(self, input_size: int) -> None:
        super().__init__()
        
        assert isinstance(input_size, int)
        assert input_size >= 2
        self.input_size: int = input_size
        
        self.nb_trainable_params: int = 0
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.input_size})"
    
    def forward_propagation(
            self,
            input_data: np.ndarray,
            training: bool = True,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Forward propagation of the Input layer
        
        Simply returns the input
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_forward_propagation_inputs(input_data, training)
        self.input  = input_data
        
        self.output = self.input
        
        if enable_checks:
            check_dtype(self.output, utils.DEFAULT_DATATYPE)
        return self.output
    
    def backward_propagation(
            self,
            output_gradient: np.ndarray,
            learning_rate: float,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Backward propagation of the Input layer
        
        NB : Here, `learning_rate` is not used because there are no trainable
             parameters
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_backward_propagation_inputs(output_gradient, learning_rate)
        
        input_gradient = output_gradient
        
        if enable_checks:
            check_dtype(input_gradient, utils.DEFAULT_DATATYPE)
        return input_gradient


##############################################################################


class DenseLayer(Layer):
    """
    Dense (i.e. fully connected) layer class
    """
    def __init__(self, nb_neurons: int, seed: Optional[int] = None) -> None:
        super().__init__()
        
        assert isinstance(nb_neurons, int)
        assert nb_neurons >= 2
        self.output_size: int = nb_neurons
        
        assert isinstance(seed, (type(None), int))
        if seed is not None:
            assert seed >= 0
        self.seed: Union[None, int] = seed
        
        # will be initialized in the `build` method (along with
        # `self.nb_trainable_params`, that was initialized to `None` via
        # `super().__init__()`)
        self.input_size: Union[None, int] = None
        self.weights: Union[None, np.ndarray] = None
        self.biases:  Union[None, np.ndarray] = None
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.output_size})"
    
    def build(self, input_size: int) -> None:
        """
        Now that we know the input size of the layer, we can actually
        initialize/build the latter. This layer is built when it is added
        to the network (with the `Network.add` method of the "network.py" script)
        """
        assert isinstance(input_size, int)
        assert input_size >= 2
        self.input_size = input_size
        
        # ------------------------------------------------------------------ #
        
        # Using the "He initialization" (mostly because it works well with
        # ReLU-like activations)
        
        initial_var = cast(2, utils.DEFAULT_DATATYPE) / cast(self.input_size, utils.DEFAULT_DATATYPE)
        initial_std = np.sqrt(initial_var)
        check_dtype(initial_std, utils.DEFAULT_DATATYPE)
        
        np.random.seed(self.seed)
        self.weights = initial_std * np.random.randn(self.input_size, self.output_size).astype(utils.DEFAULT_DATATYPE)
        self.biases  = initial_std * np.random.randn(1, self.output_size).astype(utils.DEFAULT_DATATYPE)
        np.random.seed(None) # resetting the seed
        
        # ------------------------------------------------------------------ #
        
        # = self.input_size * self.output_size + self.output_size
        # = (self.input_size + 1) * self.output_size
        self.nb_trainable_params = int(self.weights.size + self.biases.size) # we want it to be an `int`, not a `np.int_`
    
    def forward_propagation(
            self,
            input_data: np.ndarray,
            training: bool = True,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Forward propagation of the Dense layer
        
        Applies the weights and biases to the input, and returns the
        corresponding output
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_forward_propagation_inputs(input_data, training)
        self.input  = input_data
        
        self.output = self.input @ self.weights + self.biases
        
        if enable_checks:
            check_dtype(self.output, utils.DEFAULT_DATATYPE)
        return self.output
    
    def backward_propagation(
            self,
            output_gradient: np.ndarray,
            learning_rate: float,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Backward propagation of the Dense layer
        
        Computes dE/dW and dE/dB for a given output_gradient=dE/dY, updates
        the weights and biases, and returns the input_gradient=dE/dX
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            learning_rate = self._validate_backward_propagation_inputs(
                output_gradient,
                learning_rate
            )
        else:
            learning_rate = cast(learning_rate, utils.DEFAULT_DATATYPE)
        
        input_gradient = output_gradient @ self.weights.T # = dE/dX
        check_dtype(input_gradient, utils.DEFAULT_DATATYPE)
        
        batch_size = output_gradient.shape[0]
        averaging_factor = cast(1, utils.DEFAULT_DATATYPE) / cast(batch_size, utils.DEFAULT_DATATYPE)
        
        # gradient averaging for batch processing
        weights_gradient = averaging_factor * self.input.T @ output_gradient # = dE/dW
        biases_gradient = np.mean(output_gradient, axis=0, keepdims=True)    # = dE/dB
        
        if enable_checks:
            check_dtype(weights_gradient, utils.DEFAULT_DATATYPE)
            check_dtype(biases_gradient,  utils.DEFAULT_DATATYPE)
        
        # updating the trainable parameters (using gradient descent)
        self.weights -= learning_rate * weights_gradient
        self.biases  -= learning_rate * biases_gradient
        
        if enable_checks:
            check_dtype(self.weights, utils.DEFAULT_DATATYPE)
            check_dtype(self.biases,  utils.DEFAULT_DATATYPE)
        
        return input_gradient


##############################################################################


class ActivationLayer(Layer):
    """
    Activation layer class
    """
    
    # class variable
    AVAILABLE_ACTIVATIONS: dict[str, tuple[Callable, Callable]] = {
        "relu"       : (ReLU,       ReLU_prime),
        "leaky_relu" : (leaky_ReLU, leaky_ReLU_prime),
        "tanh"       : (tanh,       tanh_prime),
        "softmax"    : (softmax,    softmax_prime),
        "sigmoid"    : (sigmoid,    sigmoid_prime)
    }
    
    def __init__(
            self,
            activation_name: str,
            **kwargs: float # for `leaky_ReLU_coeff`
        ) -> None:
        
        super().__init__()
        
        # checking the validity of the specified activation name
        assert isinstance(activation_name, str)
        activation_name = activation_name.strip().lower()
        if activation_name not in ActivationLayer.AVAILABLE_ACTIVATIONS:
            raise ValueError(f"ActivationLayer.__init__ - Unrecognized activation name : \"{activation_name}\" (possible activation names : {list_to_string(list(ActivationLayer.AVAILABLE_ACTIVATIONS))})")
        
        self.activation_name: str = activation_name
        activations: tuple[Callable, Callable] = ActivationLayer.AVAILABLE_ACTIVATIONS[self.activation_name]
        self.activation:       Callable = activations[0]
        self.activation_prime: Callable = activations[1]
        
        if self.activation_name == "leaky_relu":
            default_leaky_ReLU_coeff = 0.01
            leaky_ReLU_coeff = kwargs.get("leaky_ReLU_coeff", default_leaky_ReLU_coeff)
            
            _validate_leaky_ReLU_coeff(leaky_ReLU_coeff)
            
            self.activation_kwargs: dict[str, float] = {
                "leaky_ReLU_coeff" : leaky_ReLU_coeff
            }
        else:
            self.activation_kwargs: dict[str, float] = {}
        
        # NB : Since the softmax activation only applies to VECTORS (and not
        #      scalars), the backpropagation formula won't be the same as the other
        #      activations. Essentially, the element-wise multiplication becomes
        #      an actual matrix multiplication (cf. the `backward_propagation` method)
        self._is_softmax: bool = (self.activation_name == "softmax")
        
        self.nb_trainable_params: int = 0
    
    def __str__(self) -> str:
        if self.activation_name == "leaky_relu":
            leaky_ReLU_coeff = self.activation_kwargs["leaky_ReLU_coeff"]
            precision_leaky_ReLU_coeff = max(2, count_nb_decimals_places(leaky_ReLU_coeff))
            extra_info = f", {leaky_ReLU_coeff:.{precision_leaky_ReLU_coeff}f}"
        else:
            extra_info = ""
        return f"{self.__class__.__name__}(\"{self.activation_name}\"{extra_info})"
    
    def forward_propagation(
            self,
            input_data: np.ndarray,
            training: bool = True,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Forward propagation of the Activation layer
        
        Returns the activated input
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_forward_propagation_inputs(input_data, training)
        self.input  = input_data
        
        self.output = self.activation(
            self.input,
            **self.activation_kwargs,
            enable_checks=enable_checks
        )
        
        if enable_checks:
            check_dtype(self.output, utils.DEFAULT_DATATYPE)
        return self.output
    
    def backward_propagation(
            self,
            output_gradient: np.ndarray,
            learning_rate: float,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Backward propagation of the Activation layer
        
        NB : Here, `learning_rate` is not used because there are no trainable
             parameters
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_backward_propagation_inputs(output_gradient, learning_rate)
        
        activation_prime_of_input = self.activation_prime(
            self.input,
            **self.activation_kwargs,
            enable_checks=enable_checks
        )
        
        if not(self._is_softmax):
            # element-wise multiplication
            input_gradient = output_gradient * activation_prime_of_input
        else:
            batch_size = output_gradient.shape[0]
            
            input_gradient = np.zeros(output_gradient.shape, dtype=output_gradient.dtype)
            for batch_sample_index in range(batch_size):
                # matrix multiplication (NOT element-wise multiplication)
                input_gradient[batch_sample_index, :] = output_gradient[batch_sample_index, :] @ activation_prime_of_input[batch_sample_index]
            
            """
            The previous code block is completely equivalent to the following line :
            
            input_gradient = (output_gradient.reshape(batch_size, 1, output_gradient.shape[1]) @ activation_prime_of_input).reshape(output_gradient.shape)
            
            --> Basically, we're using 3D matrix multiplication tricks to make
                the computations a bit more compact !
            """
        
        if enable_checks:
            check_dtype(input_gradient, utils.DEFAULT_DATATYPE)
        return input_gradient


##############################################################################


class BatchNormLayer(Layer):
    """
    BatchNorm regularization layer class
    """
    def __init__(self) -> None:
        super().__init__()
        
        # generic type representing the global datatype
        Float = np.dtype(utils.DEFAULT_DATATYPE).type
        
        # initializing the trainable parameters
        self.gamma: Float = cast(1, utils.DEFAULT_DATATYPE)
        self.beta:  Float = cast(0, utils.DEFAULT_DATATYPE)
        
        self.nb_trainable_params: int = 2
        
        # initializing the non-trainable parameters
        self.moving_mean: Float = cast(0, utils.DEFAULT_DATATYPE)
        self.moving_var:  Float = cast(1, utils.DEFAULT_DATATYPE)
        self.moving_std: Union[None, Float] = None
        
        # by default
        self.momentum: Float = cast(0.99, utils.DEFAULT_DATATYPE)
        assert (self.momentum > 0) and (self.momentum < 1)
        
        one: Float = cast(1, utils.DEFAULT_DATATYPE)
        self.inverse_momentum: Float = one - self.momentum
        check_dtype(self.inverse_momentum, utils.DEFAULT_DATATYPE)
        
        # by default (used for numerical stability)
        self.epsilon: Float = cast(1e-5, utils.DEFAULT_DATATYPE)
        assert (self.epsilon > 0) and (self.epsilon < 1e-2)
        
        # initializing the cached input data (to speed up the backward propagation)
        self.input_std:        Union[None, np.ndarray] = None
        self.normalized_input: Union[None, np.ndarray] = None
    
    def forward_propagation(
            self,
            input_data: np.ndarray,
            training: bool = True,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Forward propagation of the BatchNorm layer
        
        Returns the normalized (and rescaled) input along the 1st axis, i.e.
        along the batches/rows
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_forward_propagation_inputs(input_data, training)
        self.input = input_data
        
        if training:
            input_mean = self.input.mean(axis=1, keepdims=True)
            centered_input = self.input - input_mean
            input_variance = self.input.var(axis=1, keepdims=True)
            self.input_std = np.sqrt(input_variance + self.epsilon)
            
            # actually normalizing/standardizing the input
            self.normalized_input = centered_input / self.input_std
            
            # updating the non-trainable parameters
            self.moving_mean = self.momentum * self.moving_mean + self.inverse_momentum * np.mean(input_mean)
            self.moving_var  = self.momentum * self.moving_var  + self.inverse_momentum * np.mean(input_variance)
            
            if enable_checks:
                check_dtype(self.moving_mean, utils.DEFAULT_DATATYPE)
                check_dtype(self.moving_var,  utils.DEFAULT_DATATYPE)
        else:
            self.moving_std = np.sqrt(self.moving_var + self.epsilon)
            if enable_checks:
                check_dtype(self.moving_std, utils.DEFAULT_DATATYPE)
            
            self.normalized_input = (self.input - self.moving_mean) / self.moving_std
        
        if enable_checks:
            check_dtype(self.normalized_input, utils.DEFAULT_DATATYPE)
        
        # rescaling the normalized input
        self.output = self.gamma * self.normalized_input + self.beta
        
        if enable_checks:
            check_dtype(self.output, utils.DEFAULT_DATATYPE)
        return self.output
    
    def backward_propagation(
            self,
            output_gradient: np.ndarray,
            learning_rate: float,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Backward propagation of the BatchNorm layer
        
        Computes dE/d_gamma and dE/d_beta for a given output_gradient=dE/dY,
        updates gamma and beta, and returns the input_gradient=dE/dX
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            learning_rate = self._validate_backward_propagation_inputs(
                output_gradient,
                learning_rate
            )
        else:
            learning_rate = cast(learning_rate, utils.DEFAULT_DATATYPE)
        
        output_gradient_mean = output_gradient.mean(axis=1, keepdims=True)
        centered_output_gradient = output_gradient - output_gradient_mean
        
        # this variable is just defined to simplify the computations of the
        # input gradient
        intermediate_mean = np.mean(output_gradient * self.normalized_input, axis=1, keepdims=True)
        
        input_gradient = self.gamma * (centered_output_gradient - intermediate_mean * self.normalized_input) / self.input_std
        if enable_checks:
            check_dtype(input_gradient, utils.DEFAULT_DATATYPE)
        
        # gradient averaging for batch processing
        gamma_gradient = np.mean(np.sum(output_gradient * self.normalized_input, axis=1))
        beta_gradient  = np.mean(np.sum(output_gradient, axis=1))
        
        if enable_checks:
            check_dtype(gamma_gradient, utils.DEFAULT_DATATYPE)
            check_dtype(beta_gradient,  utils.DEFAULT_DATATYPE)
        
        # updating the trainable parameters (using gradient descent)
        self.gamma -= learning_rate * gamma_gradient
        self.beta  -= learning_rate * beta_gradient
        
        if enable_checks:
            check_dtype(self.gamma, utils.DEFAULT_DATATYPE)
            check_dtype(self.beta,  utils.DEFAULT_DATATYPE)
        
        return input_gradient


##############################################################################


class DropoutLayer(Layer):
    """
    Dropout regularization layer class
    """
    def __init__(self, dropout_rate: float, seed: Optional[int] = None) -> None:
        super().__init__()
        
        assert isinstance(dropout_rate, float)
        assert (dropout_rate > 0) and (dropout_rate < 1)
        self.dropout_rate: float = dropout_rate
        
        assert isinstance(seed, (type(None), int))
        if seed is not None:
            assert seed >= 0
        self.seed: Union[None, int] = seed
        
        # generic type representing the global datatype
        Float = np.dtype(utils.DEFAULT_DATATYPE).type
        
        # all the non-deactivated values will be scaled up by this factor (by default)
        one: Float = cast(1, utils.DEFAULT_DATATYPE)
        self.scaling_factor: Float = one / (one - cast(self.dropout_rate, utils.DEFAULT_DATATYPE))
        check_dtype(self.scaling_factor, utils.DEFAULT_DATATYPE)
        
        self.nb_trainable_params = 0
        
        # initializing the dropout matrix
        self.dropout_matrix: Union[None, np.ndarray] = None
    
    def __str__(self) -> str:
        precision_dropout_rate = max(2, count_nb_decimals_places(self.dropout_rate))
        return f"{self.__class__.__name__}({self.dropout_rate:.{precision_dropout_rate}f})"
    
    def generate_random_dropout_matrix(
            self,
            shape: tuple[int, int],
            dtype: Union[str, type, np.dtype],
            enable_checks: bool = True
        ):
        """
        Returns a "dropout matrix" with the specified shape and datatype. For each
        row of that matrix, the values have a probability of `self.dropout_rate`
        to be deactivated (i.e. set to zero). All the non-deactivated values
        will be set to `self.scaling_factor` (i.e. 1 / (1 - self.dropout_rate))
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            assert isinstance(shape, tuple)
            assert len(shape) == 2
            dtype = _validate_numpy_dtype(dtype)
        
        batch_size, output_size = shape
        
        dropout_matrix = self.scaling_factor * np.ones(shape, dtype=dtype)
        nb_of_values_to_drop_per_input_sample = int(round(self.dropout_rate * output_size))
        
        if nb_of_values_to_drop_per_input_sample > 0:
            choices_for_dropped_indices = np.arange(output_size)
            
            np.random.seed(self.seed)
            for batch_sample_index in range(batch_size):
                indices_of_randomly_dropped_values = np.random.choice(
                    choices_for_dropped_indices,
                    size=(nb_of_values_to_drop_per_input_sample, ),
                    replace=False
                )
                dropout_matrix[batch_sample_index, indices_of_randomly_dropped_values] = 0
            np.random.seed(None) # resetting the seed
            
            # updating the value of the seed such that the generated dropout
            # matrices aren't the same at each forward propagation (during the
            # training phase)
            if self.seed is not None:
                self.seed += 55 # the chosen increment value is arbitrary
        
        return dropout_matrix
    
    def forward_propagation(
            self,
            input_data: np.ndarray,
            training: bool = True,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Forward propagation of the Dropout layer
        
        Returns the input with randomly deactivated values
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_forward_propagation_inputs(input_data, training)
        self.input = input_data
        
        if training:
            # NB : The dropout matrix is randomly re-generated from scratch
            #      every time the forwarding method is called (during the
            #      training phase)
            self.dropout_matrix  = self.generate_random_dropout_matrix(
                shape=self.input.shape,
                dtype=self.input.dtype,
                enable_checks=enable_checks
            )
            self.output =  self.input * self.dropout_matrix
        else:
            # the "dropout process" does NOT apply during the validation and
            # testing phases
            self.output = self.input
        
        if enable_checks:
            check_dtype(self.output, utils.DEFAULT_DATATYPE)
        return self.output
    
    def backward_propagation(
            self,
            output_gradient: np.ndarray,
            learning_rate: float,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Backward propagation of the Dropout layer
        
        NB : Here, `learning_rate` is not used because there are no trainable
             parameters
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_backward_propagation_inputs(output_gradient, learning_rate)
        
        input_gradient = output_gradient * self.dropout_matrix
        
        if enable_checks:
            check_dtype(input_gradient, utils.DEFAULT_DATATYPE)
        return input_gradient

