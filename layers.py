# -*- coding: utf-8 -*-

"""
Script defining the main layer classes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pickle import dumps, loads
from typing import Union, Optional, Callable

import numpy as np

import utils
from utils import (
    cast,
    check_dtype,
    list_to_string,
    generate_unique_ID,
    count_nb_decimals_places,
    _validate_numpy_dtype,
    _validate_leaky_ReLU_coeff
)

from optimizers import (
    Optimizer,
    SgdOptimizer,
    AdamOptimizer,
    RMSpropOptimizer
)

from regularizers import Regularizer, L1, L2, L1_L2

from activations import (
    ReLU,        ReLU_prime,
    leaky_ReLU,  leaky_ReLU_prime,
    tanh,        tanh_prime,
    softmax,     softmax_prime,
    log_softmax, log_softmax_prime,
    sigmoid,     sigmoid_prime
)


##############################################################################


class Layer(ABC):
    """
    Base (abstract) layer class
    """
    
    # ---------------------------------------------------------------------- #
    
    # Defining the class variables
    
    AVAILABLE_OPTIMIZERS: dict[str, Optimizer] = {
        "sgd"     : SgdOptimizer,
        "adam"    : AdamOptimizer,
        "rmsprop" : RMSpropOptimizer
    }
    
    # used for the `__call__` API
    TEMPORARY_NETWORK_LAYERS: list[Layer] = []
    
    # ---------------------------------------------------------------------- #
    
    def __init__(self) -> None:
        self.input:  Union[None, np.ndarray] = None
        self.output: Union[None, np.ndarray] = None
        
        self.nb_trainable_params: Union[None, int] = None
        
        # defined for convenience purposes
        self._zero: float = cast(0, utils.DEFAULT_DATATYPE)
        self._one:  float = cast(1, utils.DEFAULT_DATATYPE)
        self._two:  float = cast(2, utils.DEFAULT_DATATYPE)
        
        # variables related to L1/L2 regularization
        self._has_regularizer: bool = False
        self.regularizer: Optional[Regularizer] = None
        self.loss_leftovers: float = self._zero
        self._L1_coeff: Optional[float] = None
        self._L2_coeff: Optional[float] = None
        self._L2_scaling_factor: Optional[float] = None
        
        # these variables will be set by the `Layer.set_optimizer` method
        self.optimizer_name: Union[None, str] = None
        self._learning_rate: Union[None, float] = None
        self._optimizer: Union[None, Optimizer] = None
        self._optimize_weights: Union[None, Callable] = None
        
        # If this boolean is set to `True`, then all the trainable and
        # non-trainable parameters of the current Layer instance will be
        # frozen. This feature can be used for Transfer Learning purposes
        # (for example)
        self._is_frozen: bool = False
        
        # the name of the current Layer instance will be given once it is
        # actually added to the network (using the `Network.add` method of
        # the "network.py" script)
        self._name: Union[None, str] = None
        
        self._unique_ID: int = generate_unique_ID()
    
    def __str__(self) -> str:
        # default string representation of the layer classes (most of the
        # time their `__str__` method will override this one)
        return f"{self.__class__.__name__}()"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __call__(self, layer: Layer) -> Layer:
        """
        Adds the current Layer instance (i.e. `self`) to the list of layers
        that will be used to build the network's architecture using the
        `__call__` API (i.e. `Layer.TEMPORARY_NETWORK_LAYERS`)
        
        See the `Network.__call__` method (of the "network.py" script) for a
        concrete example
        """
        assert issubclass(type(layer), Layer)
        
        if len(Layer.TEMPORARY_NETWORK_LAYERS) == 0:
            Layer.TEMPORARY_NETWORK_LAYERS.append(layer)
        
        copy_of_current_layer_instance = self.copy()
        Layer.TEMPORARY_NETWORK_LAYERS.append(copy_of_current_layer_instance)
        
        # the returned copy has to be the *same* as the one that was just
        # stored in the `Layer.TEMPORARY_NETWORK_LAYERS` list (just in case
        # it's the output layer)
        return copy_of_current_layer_instance
    
    @staticmethod
    def clear_temporary_network_layers() -> None:
        """
        Simply sets `Layer.TEMPORARY_NETWORK_LAYERS` to the empty list.
        This method is only called by the `Network.__call__` method (of the
        "network.py" script)
        """
        Layer.TEMPORARY_NETWORK_LAYERS = []
    
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
    
    def _add_regularizer(self, regularizer: Regularizer):
        """
        Adds a L1/L2 regularizer to the current layer
        """
        assert isinstance(regularizer, (type(None), Regularizer))
        
        if (regularizer is not None) and (self.regularizer is None):
            assert type(regularizer) != Regularizer
            
            if hasattr(regularizer, "L1_coeff"):
                self._L1_coeff = cast(regularizer.L1_coeff, utils.DEFAULT_DATATYPE)
            if hasattr(regularizer, "L2_coeff"):
                self._L2_coeff = cast(regularizer.L2_coeff, utils.DEFAULT_DATATYPE)
                self._L2_scaling_factor = self._L2_coeff / self._two
                check_dtype(self._L2_scaling_factor, utils.DEFAULT_DATATYPE)
            
            assert (self._L1_coeff is not None) or (self._L2_coeff is not None)
            self.regularizer = regularizer
            self._has_regularizer = True
    
    def set_optimizer(
            self,
            optimizer_name: str,
            *,
            learning_rate: float = 0.001
        ) -> None:
        """
        Sets the optimizer of the current layer to the specified optimizer.
        The optimizer name is case insensitive
        """
        # ------------------------------------------------------------------- #
        
        # checking the specified arguments
        
        assert isinstance(optimizer_name, str)
        assert len(optimizer_name.strip()) > 0
        optimizer_name = optimizer_name.strip().lower()
        
        if optimizer_name not in Layer.AVAILABLE_OPTIMIZERS:
            raise ValueError(f"{self.__class__.__name__}.set_optimizer - Unrecognized optimizer name : \"{optimizer_name}\" (available optimizer names : {list_to_string(list(Layer.AVAILABLE_OPTIMIZERS))})")
        
        assert isinstance(learning_rate, float)
        assert (learning_rate > 0) and (learning_rate < 1)
        
        # ------------------------------------------------------------------- #
        
        self.optimizer_name = optimizer_name
        self._learning_rate = learning_rate
        
        optimizer_class = Layer.AVAILABLE_OPTIMIZERS[self.optimizer_name]
        self._optimizer = optimizer_class(self._learning_rate, regularizer=self.regularizer)
        
        self._optimize_weights = self._optimizer.optimize_weights
    
    def _check_if_optimizer_is_set(self) -> None:
        """
        Simply checks if an optimizer has been set or not (on the called layer)
        """
        assert self.optimizer_name is not None
    
    @abstractmethod
    def forward_propagation(
            self,
            input_data: np.ndarray,
            *,
            training: bool = False,
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
        assert input_data.ndim == 2
        check_dtype(input_data, utils.DEFAULT_DATATYPE)
        
        assert isinstance(training, bool)
    
    @abstractmethod
    def backward_propagation(
            self,
            output_gradient: np.ndarray,
            *,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Computes dE/dX for a given dE/dY, and updates the trainable parameters
        if there are any
        """
        pass
    
    def _validate_backward_propagation_input(
            self,
            output_gradient: np.ndarray
        ) -> None:
        """
        Checks if `output_gradient` is valid or not
        """
        assert isinstance(output_gradient, np.ndarray)
        assert output_gradient.ndim == 2
        check_dtype(output_gradient, utils.DEFAULT_DATATYPE)
    
    def freeze(self) -> None:
        """
        Freezes the trainable and non-trainable parameters of the current
        Layer instance (if it has any)
        
        This method can be used for Transfer Learning purposes for example
        """
        self._is_frozen = True
    
    def _pickle(self) -> bytes:
        """
        Returns the pickled/serialized version (i.e. a byte representation)
        of the current Layer instance
        """
        variables_of_current_layer = self.__dict__.copy()
        
        # the `regularizer`, `_optimizer` and `_optimize_weights` attributes
        # aren't pickleable/serializable
        variables_of_current_layer.pop("regularizer")
        variables_of_current_layer.pop("_optimizer")
        variables_of_current_layer.pop("_optimize_weights")
        
        pickled_layer = dumps(variables_of_current_layer)
        assert isinstance(pickled_layer, bytes)
        assert len(pickled_layer) > 0
        
        return pickled_layer
    
    @classmethod
    def _load(cls, pickled_layer: bytes) -> Layer:
        """
        Loads (a copy of) the current Layer instance, directly from the
        specified `pickled_layer` argument. This method is the inverse of the
        `Layer._pickle` method
        """
        assert isinstance(pickled_layer, bytes)
        assert len(pickled_layer) > 0
        
        loaded_layer_as_dict = loads(pickled_layer)
        assert isinstance(loaded_layer_as_dict, dict)
        assert len(loaded_layer_as_dict) > 0
        
        # initializing the loaded layer
        loaded_layer = cls.__new__(cls)
        
        for variable_name, variable in loaded_layer_as_dict.items():
            assert isinstance(variable_name, str)
            setattr(loaded_layer, variable_name, variable)
        
        # setting the `regularizer` attribute
        L1_coeff = loaded_layer._L1_coeff
        L2_coeff = loaded_layer._L2_coeff
        if (L1_coeff is None) and (L2_coeff is None):
            regularizer = None
        elif (L1_coeff is not None) and (L2_coeff is None):
            regularizer = L1(L1_coeff)
        elif (L1_coeff is None) and (L2_coeff is not None):
            regularizer = L2(L2_coeff)
        else:
            # in this case, `L1_coeff` and `L2_coeff` are both different
            # from `None`
            regularizer = L1_L2(L1_coeff, L2_coeff)
        loaded_layer.regularizer = regularizer
        
        # setting the `_optimizer` and `_optimize_weights` attributes
        optimizer_name = loaded_layer.optimizer_name
        learning_rate  = loaded_layer._learning_rate
        if learning_rate is not None:
            assert optimizer_name is not None
            loaded_layer.set_optimizer(optimizer_name, learning_rate=learning_rate)
        else:
            assert optimizer_name is None
            loaded_layer._optimizer = None
            loaded_layer._optimize_weights = None
        
        loaded_layer._unique_ID = generate_unique_ID()
        
        return loaded_layer
    
    def copy(self) -> Layer:
        """
        Returns a copy of the current Layer instance
        """
        layer_copy = self.__class__._load(self._pickle())
        
        assert type(layer_copy) == type(self)
        assert sorted(list(layer_copy.__dict__)) == sorted(list(self.__dict__))
        assert layer_copy is not self
        
        return layer_copy


##############################################################################


class MyLayer(Layer):
    """
    Template for defining a new Layer class. The `forward_propagation` and
    `backward_propagation` methods need to have these exact signatures
    """
    def __init__(self, *args, **kwargs) -> None:
        # Note that this class has to inherit from the base `Layer` class
        super().__init__()
        
        # Define all the instance variables you need (make sure your weights
        # and/or other variables are *all* cast to `utils.DEFAULT_DATATYPE`)
        ...
        
        # Manually define the total number of trainable parameters in the layer
        self.nb_trainable_params: int = ...
    
    def build(self, input_size: int) -> None:
        # Optional method used to define some instance variables if the
        # input size of the layer is unknown when the layer is instantiated
        # (and assuming the input size is a defining feature of the layer)
        pass
    
    def forward_propagation(
            self,
            input_data: np.ndarray,
            *,
            training: bool = False,
            enable_checks: bool = True
        ) -> np.ndarray:
        
        # Save the input data (since it might be needed in the backward propagation)
        self.input = input_data
        
        # Compute the output directly from the input
        self.output = ...
        
        return self.output
    
    def backward_propagation(
            self,
            output_gradient: np.ndarray,
            *,
            enable_checks: bool = True
        ) -> np.ndarray:
        
        # Compute the input gradient directly from the given output gradient
        input_gradient = ...
        
        if not(self._is_frozen):
            # Update the trainable parameters (if there are any). Uncomment the
            # following code block if needed :
            
            # self.weights_1, self.weights_2, ... = self._optimize_weights(
            #     weights=(self.weights_1, self.weights_2, ...),
            #     weight_gradients=(weight_gradient_1, weight_gradient_2, ...),
            #     enable_checks=enable_checks
            # )
            
            pass
        
        # ------------------------------------------------------------------ #
        
        # This section is only relevant if this layer has trainable parameters
        # that are interesting to regularize using the L1/L2 regularizers. In
        # that case, you'll need to call `self._add_regularizer(regularizer)`
        # at some stage before the training occurs (e.g. in the `__init__`
        # method). Here, `regularizer` has to be an instance of one of the
        # `L1`, `L2` or `L1_L2` classes
        
        if self._has_regularizer:
            # resetting the "loss leftovers" (for the L1/L2 regularizers)
            self.loss_leftovers = self._zero
            
            if self._L1_coeff is not None:
                # += L1_coeff * sum([np.sum(np.abs(weights)) for weights in list_of_all_weight_matrices])
                self.loss_leftovers += self._L1_coeff * (np.sum(np.abs(self.weights_1)) + np.sum(np.abs(self.weights_2)) + ...)
            if self._L2_coeff is not None:
                # += L2_scaling_factor * sum([np.sum(np.square(weights)) for weights in list_of_all_weight_matrices])
                self.loss_leftovers += self._L2_scaling_factor * (np.sum(np.square(self.weights_1)) + np.sum(np.square(self.weights_2)) + ...)
        
        # ------------------------------------------------------------------ #
        
        return input_gradient


"""
Then, import your layer class at the beginning of the "network.py" script,
and add it to the `Network.AVAILABLE_LAYER_TYPES` dictionary. The latter
dictionary is a (class) variable defined right before the `Network.__init__`
method (also in the "network.py" script)
"""


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
            *,
            training: bool = False,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Forward propagation of the Input layer
        
        Simply returns the input
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_forward_propagation_inputs(input_data, training)
        self.input = input_data
        
        self.output = self.input
        
        if enable_checks:
            check_dtype(self.output, utils.DEFAULT_DATATYPE)
        return self.output
    
    def backward_propagation(
            self,
            output_gradient: np.ndarray,
            *,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Backward propagation of the Input layer
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_backward_propagation_input(output_gradient)
        
        input_gradient = output_gradient
        
        if enable_checks:
            check_dtype(input_gradient, utils.DEFAULT_DATATYPE)
        return input_gradient


##############################################################################


class DenseLayer(Layer):
    """
    Dense (i.e. fully connected) layer class
    """
    def __init__(
            self,
            nb_neurons: int,
            *,
            use_biases: bool = True,
            regularizer: Optional[Regularizer] = None,
            seed: Optional[int] = None
        ) -> None:
        
        super().__init__()
        
        assert isinstance(nb_neurons, int)
        assert nb_neurons >= 2
        self.output_size: int = nb_neurons
        
        assert isinstance(use_biases, bool)
        self.use_biases = use_biases
        
        self._add_regularizer(regularizer)
        
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
        
        # used to speed up the computations a bit during backpropagation
        self._batch_size: Union[None, int] = None
        self._averaging_factor: Union[None, float] = None
    
    def __str__(self) -> str:
        extra_info = ""
        if not(self.use_biases):
            extra_info += ", use_biases=False"
        if self._is_frozen:
            extra_info += ", is_frozen=True"
        
        return f"{self.__class__.__name__}({self.output_size}{extra_info})"
    
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
        
        # Using the "He initialization" for the weights (primarily because
        # it works well with ReLU-like activations)
        
        initial_var = self._two / cast(self.input_size, utils.DEFAULT_DATATYPE)
        initial_std = np.sqrt(initial_var)
        check_dtype(initial_std, utils.DEFAULT_DATATYPE)
        
        np.random.seed(self.seed)
        self.weights = initial_std * np.random.randn(self.input_size, self.output_size).astype(utils.DEFAULT_DATATYPE)
        np.random.seed(None) # resetting the seed
        
        # ------------------------------------------------------------------ #
        
        # Initializing all the biases to zero
        
        if self.use_biases:
            self.biases = np.zeros((1, self.output_size), dtype=utils.DEFAULT_DATATYPE)
        
        # ------------------------------------------------------------------ #
        
        if self.use_biases:
            # = self.input_size * self.output_size + self.output_size
            # = (self.input_size + 1) * self.output_size
            self.nb_trainable_params = int(self.weights.size + self.biases.size) # we want it to be an `int`, not a `np.int_`
        else:
            # = self.input_size * self.output_size
            self.nb_trainable_params = int(self.weights.size) # we want it to be an `int`, not a `np.int_`
    
    def forward_propagation(
            self,
            input_data: np.ndarray,
            *,
            training: bool = False,
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
        self.input = input_data
        
        if self.use_biases:
            self.output = self.input @ self.weights + self.biases
        else:
            self.output = self.input @ self.weights
        
        if enable_checks:
            check_dtype(self.output, utils.DEFAULT_DATATYPE)
        return self.output
    
    def backward_propagation(
            self,
            output_gradient: np.ndarray,
            *,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Backward propagation of the Dense layer
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._check_if_optimizer_is_set()
            self._validate_backward_propagation_input(output_gradient)
        
        input_gradient = output_gradient @ self.weights.T
        if enable_checks:
            check_dtype(input_gradient, utils.DEFAULT_DATATYPE)
        
        if not(self._is_frozen):
            batch_size = output_gradient.shape[0]
            if batch_size != self._batch_size:
                self._batch_size = batch_size
                self._averaging_factor = self._one / cast(batch_size, utils.DEFAULT_DATATYPE)
            
            # gradient averaging for batch processing
            weights_gradient = self._averaging_factor * self.input.T @ output_gradient
            if self.use_biases:
                biases_gradient = np.mean(output_gradient, axis=0, keepdims=True)
            
            # updating the trainable parameters
            if self.use_biases:
                self.weights, self.biases = self._optimize_weights(
                    weights=(self.weights, self.biases),
                    weight_gradients=(weights_gradient, biases_gradient),
                    enable_checks=enable_checks
                )
            else:
                self.weights, = self._optimize_weights(
                    weights=(self.weights,),
                    weight_gradients=(weights_gradient,),
                    enable_checks=enable_checks
                )
            
            # -------------------------------------------------------------- #
            
            # Part related to L1/L2 regularization
            
            if self._has_regularizer:
                # resetting the "loss leftovers"
                self.loss_leftovers = self._zero
                
                if self._L1_coeff is not None:
                    self.loss_leftovers += self._L1_coeff * np.sum(np.abs(self.weights))
                    if self.use_biases:
                        self.loss_leftovers += self._L1_coeff * np.sum(np.abs(self.biases))
                
                if self._L2_coeff is not None:
                    self.loss_leftovers += self._L2_scaling_factor * np.sum(np.square(self.weights))
                    if self.use_biases:
                        self.loss_leftovers += self._L2_scaling_factor * np.sum(np.square(self.biases))
            
            # -------------------------------------------------------------- #
        
        return input_gradient


##############################################################################


class ActivationLayer(Layer):
    """
    Activation layer class
    """
    
    # class variable
    AVAILABLE_ACTIVATIONS: dict[str, tuple[Callable, Callable]] = {
        "relu"        : (ReLU,        ReLU_prime),
        "leaky_relu"  : (leaky_ReLU,  leaky_ReLU_prime),
        "prelu"       : (leaky_ReLU,  leaky_ReLU_prime),
        "tanh"        : (tanh,        tanh_prime),
        "softmax"     : (softmax,     softmax_prime),
        "log_softmax" : (log_softmax, log_softmax_prime),
        "sigmoid"     : (sigmoid,     sigmoid_prime)
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
        self._activation, self._activation_prime = activations
        
        # to prevent typos
        self._key_name_for_leaky_ReLU_coeff: str = "leaky_ReLU_coeff"
        
        self.activation_kwargs: dict[str, float] = {}
        if self.activation_name == "leaky_relu":
            default_leaky_ReLU_coeff = 0.01
            
            leaky_ReLU_coeff = kwargs.get(self._key_name_for_leaky_ReLU_coeff, default_leaky_ReLU_coeff)
            _validate_leaky_ReLU_coeff(leaky_ReLU_coeff)
            
            self.activation_kwargs[self._key_name_for_leaky_ReLU_coeff] = leaky_ReLU_coeff
        
        # NB : Since the softmax activation only applies to VECTORS (and not
        #      scalars), the backpropagation formula won't be the same as the other
        #      activations. Essentially, the element-wise multiplication becomes
        #      an actual matrix multiplication (cf. the `backward_propagation` method).
        #      The same goes for the log-softmax activation
        self._is_softmax: bool = (self.activation_name in ["softmax", "log_softmax"])
        
        # The Parameterized ReLU activation (or "PReLU") is basically a leaky
        # ReLU activation where the "leaky ReLU coefficient" is unbounded,
        # and is a trainable parameter
        self._is_prelu: bool = (self.activation_name == "prelu")
        
        if self._is_prelu:
            # By default, the only trainable parameter of the PReLU activation
            # layer (i.e. `self.coeff`) will be initialized to zero. It can
            # be interpreted as some  kind of "unbounded leaky ReLU coefficient
            # varying over time"
            self.coeff: Union[None, float] = self._zero
            self.nb_trainable_params: int = 1
        else:
            self.coeff: Union[None, float] = None
            self.nb_trainable_params: int = 0
        
        if self.activation_name in ["tanh", "softmax", "log_softmax", "sigmoid"]:
            self._reuse_activation_output_in_backprop: bool = True
        else:
            self._reuse_activation_output_in_backprop: bool = False
    
    def __str__(self) -> str:
        extra_info = ""
        if self.activation_name == "leaky_relu":
            leaky_ReLU_coeff = self.activation_kwargs[self._key_name_for_leaky_ReLU_coeff]
            precision_leaky_ReLU_coeff = max(2, count_nb_decimals_places(leaky_ReLU_coeff))
            str_leaky_ReLU_coeff = f"{leaky_ReLU_coeff:.{precision_leaky_ReLU_coeff}f}"
            extra_info += f", coeff={str_leaky_ReLU_coeff}"
        elif self._is_prelu and self._is_frozen:
            extra_info += ", is_frozen=True"
        
        return f"{self.__class__.__name__}(\"{self.activation_name}\"{extra_info})"
    
    def forward_propagation(
            self,
            input_data: np.ndarray,
            *,
            training: bool = False,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Forward propagation of the Activation layer
        
        Returns the activated input
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_forward_propagation_inputs(input_data, training)
        self.input = input_data
        
        if self._is_prelu:
            self.activation_kwargs[self._key_name_for_leaky_ReLU_coeff] = self.coeff
        
        self.output = self._activation(
            self.input,
            **self.activation_kwargs,
            enable_checks=False
        )
        
        if enable_checks:
            check_dtype(self.output, utils.DEFAULT_DATATYPE)
        return self.output
    
    def backward_propagation(
            self,
            output_gradient: np.ndarray,
            *,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Backward propagation of the Activation layer
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_backward_propagation_input(output_gradient)
            
            if self._is_prelu:
                assert np.allclose(self.activation_kwargs[self._key_name_for_leaky_ReLU_coeff], self.coeff)
        
        if self._reuse_activation_output_in_backprop:
            used_input = self.output
        else:
            used_input = self.input
        
        activation_prime_of_input = self._activation_prime(
            used_input,
            **self.activation_kwargs,
            input_is_activation_output=self._reuse_activation_output_in_backprop,
            enable_checks=False
        )
        
        if not(self._is_softmax):
            # element-wise multiplication
            input_gradient = output_gradient * activation_prime_of_input
            
            if self._is_prelu and not(self._is_frozen):
                coeff_gradient = np.sum(output_gradient * np.minimum(self.input, 0))
                
                # updating the (only) trainable parameter of the PReLU layer
                self.coeff, = self._optimize_weights(
                    weights=(self.coeff,),
                    weight_gradients=(coeff_gradient,),
                    enable_checks=enable_checks
                )
        else:
            # matrix multiplication (NOT element-wise multiplication)
            input_gradient = np.squeeze(np.expand_dims(output_gradient, axis=1) @ activation_prime_of_input)
        
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
        
        # initializing the trainable parameters
        self.gamma: float = self._one
        self.beta:  float = self._zero
        
        self.nb_trainable_params: int = 2
        
        # initializing the non-trainable parameters
        self.moving_mean: float = self._zero
        self.moving_var:  float = self._one
        
        # used to speed up the computations a bit during the forward
        # propagation (during validation and testing)
        self._moving_var_was_changed: bool = False
        
        self.moving_std: float = self._one
        
        # by default
        self.momentum: float = cast(0.99, utils.DEFAULT_DATATYPE)
        assert (self.momentum > 0) and (self.momentum < 1)
        
        self._momentum_inverse: float = self._one - self.momentum
        check_dtype(self._momentum_inverse, utils.DEFAULT_DATATYPE)
        
        # by default (used for numerical stability)
        self.epsilon: float = cast(1e-5, utils.DEFAULT_DATATYPE)
        assert (self.epsilon > 0) and (self.epsilon < 1e-2)
        
        # initializing the cached input data (which is used to slightly speed
        # up the backward propagation)
        self.input_std:        Union[None, np.ndarray] = None
        self.normalized_input: Union[None, np.ndarray] = None
    
    def __str__(self) -> str:
        extra_info = ""
        if self._is_frozen:
            extra_info += "is_frozen=True"
        
        return f"{self.__class__.__name__}({extra_info})"
    
    def forward_propagation(
            self,
            input_data: np.ndarray,
            *,
            training: bool = False,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Forward propagation of the BatchNorm layer
        
        Returns the standardized (and rescaled) input along the first axis, i.e.
        along the batch samples (i.e. along the rows)
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_forward_propagation_inputs(input_data, training)
        self.input = input_data
        
        if training and not(self._is_frozen):
            input_mean = self.input.mean(axis=1, keepdims=True)
            centered_input = self.input - input_mean
            input_variance = self.input.var(axis=1, keepdims=True)
            self.input_std = np.sqrt(input_variance + self.epsilon)
            
            # actually normalizing/standardizing the input
            self.normalized_input = centered_input / self.input_std
            
            # updating the non-trainable parameters
            self.moving_mean = self.momentum * self.moving_mean + self._momentum_inverse * np.mean(input_mean)
            self.moving_var  = self.momentum * self.moving_var  + self._momentum_inverse * np.mean(input_variance)
            self._moving_var_was_changed = True
            
            if enable_checks:
                check_dtype(self.moving_mean, utils.DEFAULT_DATATYPE)
                check_dtype(self.moving_var,  utils.DEFAULT_DATATYPE)
        else:
            if self._moving_var_was_changed:
                self.moving_std = np.sqrt(self.moving_var + self.epsilon)
                if enable_checks:
                    check_dtype(self.moving_std, utils.DEFAULT_DATATYPE)
                self._moving_var_was_changed = False
            
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
            *,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Backward propagation of the BatchNorm layer
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._check_if_optimizer_is_set()
            self._validate_backward_propagation_input(output_gradient)
        
        output_gradient_mean = output_gradient.mean(axis=1, keepdims=True)
        centered_output_gradient = output_gradient - output_gradient_mean
        
        # this variable is just defined to simplify the computations of the
        # input gradient
        intermediate_mean = np.mean(output_gradient * self.normalized_input, axis=1, keepdims=True)
        
        input_gradient = self.gamma * (centered_output_gradient - self.normalized_input * intermediate_mean) / self.input_std
        if enable_checks:
            check_dtype(input_gradient, utils.DEFAULT_DATATYPE)
        
        if not(self._is_frozen):
            # gradient averaging for batch processing
            gamma_gradient = np.mean(np.sum(output_gradient * self.normalized_input, axis=1))
            beta_gradient  = np.mean(np.sum(output_gradient, axis=1))
            
            # updating the trainable parameters
            self.gamma, self.beta = self._optimize_weights(
                weights=(self.gamma, self.beta),
                weight_gradients=(gamma_gradient, beta_gradient),
                enable_checks=enable_checks
            )
        
        return input_gradient


##############################################################################


class DropoutLayer(Layer):
    """
    Dropout regularization layer class
    """
    def __init__(
            self,
            dropout_rate: float,
            *,
            seed: Optional[int] = None
        ) -> None:
        
        super().__init__()
        
        assert isinstance(dropout_rate, float)
        assert (dropout_rate > 0) and (dropout_rate < 1)
        self.dropout_rate: float = dropout_rate
        
        assert isinstance(seed, (type(None), int))
        if seed is not None:
            assert seed >= 0
        self.seed: Union[None, int] = seed
        
        # all the non-deactivated values will be scaled up by this factor (by default)
        self.scaling_factor: float = self._one / (self._one - cast(self.dropout_rate, utils.DEFAULT_DATATYPE))
        check_dtype(self.scaling_factor, utils.DEFAULT_DATATYPE)
        
        self.nb_trainable_params = 0
        
        # initializing the dropout matrix
        self.dropout_matrix: Union[None, np.ndarray] = None
    
    def __str__(self) -> str:
        precision_dropout_rate = max(2, count_nb_decimals_places(self.dropout_rate))
        str_dropout_rate = f"{self.dropout_rate:.{precision_dropout_rate}f}"
        
        extra_info = ""
        if self._is_frozen:
            extra_info += ", is_frozen=True"
        
        return f"{self.__class__.__name__}({str_dropout_rate}{extra_info})"
    
    def generate_random_dropout_matrix(
            self,
            shape: tuple[int, int],
            dtype: Union[str, type, np.dtype],
            *,
            enable_checks: bool = True
        ) -> np.ndarray:
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
        nb_of_values_to_drop_per_input_sample = int(round(self.dropout_rate * output_size))
        
        if nb_of_values_to_drop_per_input_sample > 0:
            dropout_matrix = np.full(shape, self.scaling_factor, dtype=dtype)
            
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
                self.seed += 34 # the chosen increment value is arbitrary
        else:
            dropout_matrix = np.ones(shape, dtype=dtype)
        
        return dropout_matrix
    
    def forward_propagation(
            self,
            input_data: np.ndarray,
            *,
            training: bool = False,
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
        
        if training and not(self._is_frozen):
            # NB : The dropout matrix is randomly re-generated from scratch
            #      every time the forwarding method is called (during the
            #      training phase)
            self.dropout_matrix = self.generate_random_dropout_matrix(
                shape=self.input.shape,
                dtype=self.input.dtype,
                enable_checks=False
            )
            self.output = self.input * self.dropout_matrix
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
            *,
            enable_checks: bool = True
        ) -> np.ndarray:
        """
        Backward propagation of the Dropout layer
        """
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            self._validate_backward_propagation_input(output_gradient)
        
        if not(self._is_frozen):
            input_gradient = output_gradient * self.dropout_matrix
        else:
            input_gradient = output_gradient
        
        if enable_checks:
            check_dtype(input_gradient, utils.DEFAULT_DATATYPE)
        return input_gradient

