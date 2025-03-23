"""
Script defining the main network class
"""

from __future__ import annotations

import os
from time import perf_counter
from pickle import dumps, loads
from gzip import open as gzip_open
from typing import Union, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt

# used to force integer ticks on the x-axis of a plot
from matplotlib.ticker import MaxNLocator

# used to print colored text in a Python console/terminal
from colorama import init, Fore, Style

import src.utils as utils
from src.utils import (
    cast,
    check_dtype,
    standardize_data,
    vector_to_categorical,
    categorical_to_vector,
    list_to_string,
    generate_unique_ID,
    progress_bar,
    is_being_run_on_jupyter_notebook,
    count_nb_decimals_places,
    format_runtime,
    _validate_label_vector,
    _validate_selected_classes,
    _validate_numpy_dtype,
    _validate_one_hot_encoded_array
)

from src.core import (
    split_data_into_batches,
    accuracy_score,
    confusion_matrix,
    train_test_split
)

from src.losses import (
    CCE, CCE_prime,
    MSE, MSE_prime
)

from src.layers import (
    Layer,
    InputLayer,
    DenseLayer,
    ActivationLayer,
    BatchNormLayer,
    DropoutLayer
)

from src.callbacks import (
    Callback,
    EarlyStoppingCallback
)


##############################################################################


class Network:
    """
    Network class
    """
    
    # ---------------------------------------------------------------------- #
    
    # Defining the class variables
    
    AVAILABLE_LAYER_TYPES: tuple[Layer] = (
        InputLayer,
        DenseLayer,
        ActivationLayer,
        BatchNormLayer,
        DropoutLayer
    )
    
    AVAILABLE_LOSSES: dict[str, tuple[Callable, Callable]] = {
        "cce" : (CCE, CCE_prime), # CCE = Categorical Cross-Entropy
        "mse" : (MSE, MSE_prime)  # MSE = Mean Squared Error
    }
    
    DEFAULT_SAVED_PLOTS_FOLDER_NAME = "saved_plots"
    
    DEFAULT_SAVED_NETWORKS_FOLDER_NAME = "saved_networks"
    
    # ---------------------------------------------------------------------- #
    
    def __init__(
            self,
            *,
            standardize_input_data: bool = True
        ) -> None:
        assert isinstance(standardize_input_data, bool)
        self._standardize_input_data: bool = standardize_input_data
        
        self._layers: list[Layer] = []
        
        # list containing the input/output sizes of all the layers of the
        # network (it's a list of tuples of 2 integers)
        self._io_sizes: list[tuple[int, int]] = []
        
        # this dictionary keeps track of the number of layer instances that
        # have already been added to the network (regarding each layer type)
        self._layer_counter: dict[str, int] = {}
        
        # initializing `_layer_counter`
        for layer_type in Network.AVAILABLE_LAYER_TYPES:
            layer_name = layer_type.__name__
            self._layer_counter[layer_name] = 0
        
        # for reused layers (e.g. for Transfer Learning purposes)
        self._key_name_for_reused_layers = "REUSED" # to prevent typos
        self._layer_counter[self._key_name_for_reused_layers] = 0
        
        self._output_activation_is_log_softmax: bool = False
        
        self.loss_name: Union[None, str] = None
        self._loss: Union[None, Callable] = None
        self._loss_prime: Union[None, Callable] = None
        
        self.optimizer_name: Union[None, str] = None
        self._learning_rate: Union[None, float] = None
        
        # defined for convenience purposes
        self._zero = cast(0, utils.DEFAULT_DATATYPE)
        
        self.history: Union[None, dict[str, list]] = None
        self._is_trained: bool = False
        
        self._unique_ID: int = generate_unique_ID()
    
    
    def __str__(self) -> str:
        if len(self._layers) == 0:
            return f"{self.__class__.__name__}()"
        
        # using the default summary kwargs (except for `print_summary`, which
        # has to be set to `False` here, in order to actually return a string,
        # and not a `NoneType`)
        return self.summary(print_summary=False)
    
    
    def __repr__(self) -> str:
        return str(self)
    
    
    def _set_layer_name(self, layer: Layer) -> None:
        """
        Sets the `_name` attribute of the specified layer
        """
        assert issubclass(type(layer), Layer)
        
        layer_name = layer.__class__.__name__
        layer_name_abbreviation = layer_name.lower().replace("layer", "")
        
        if layer._name is None:
            self._layer_counter[layer_name] += 1
            layer_index = self._layer_counter[layer_name]
            full_name_of_layer = f"{layer_name_abbreviation}_{layer_index}"
        else:
            # here, that means that we've (most likely) reused a layer from
            # another Network instance (e.g. for Transfer Learning purposes)
            self._layer_counter[self._key_name_for_reused_layers] += 1
            reused_layer_index = self._layer_counter[self._key_name_for_reused_layers]
            full_name_of_layer = f"{layer_name_abbreviation}_reused_{reused_layer_index}"
        
        layer._name = full_name_of_layer
    
    
    def get_layer_by_name(self, layer_name: str) -> Layer:
        """
        Returns a copy of the layer that has the specified layer name
        (e.g. "dense_1")
        """
        # checking the specified layer name
        assert isinstance(layer_name, str)
        assert len(layer_name.strip()) > 0
        layer_name = layer_name.strip().lower().replace("layer", "").replace(" ", "_")
        
        for layer in self._layers:
            assert layer._name is not None
            if layer_name == layer._name:
                return layer.copy()
        
        raise ValueError(f"Network.get_layer_by_name - The specified layer name (= \"{layer_name}\") doesn't exist !")
    
    
    def add(self, layer: Layer) -> None:
        """
        Adds a layer to the network
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified layer
        
        assert issubclass(type(layer), Layer)
        
        if not(isinstance(layer, Network.AVAILABLE_LAYER_TYPES)):
            raise TypeError(f"Network.add - Unrecognized layer type : \"{layer.__class__.__name__}\" (available layer types : {list_to_string(Network.AVAILABLE_LAYER_TYPES)})")
        
        # ------------------------------------------------------------------ #
        
        assert len(self._layers) == len(self._io_sizes)
        
        if isinstance(layer, InputLayer):
            assert len(self._layers) == 0, "\nNetwork.add - You cannot add an InputLayer if other layers have already been added to the Network !"
            input_size  = layer.input_size
        else:
            assert len(self._layers) >= 1, f"\nNetwork.add - Please add an InputLayer to the network before adding a \"{layer.__class__.__name__}\" !"
            input_size  = self._io_sizes[-1][1] # output size of the previous layer
        
        # in case a Layer instance is added to the network multiple times
        # (as some kind of "template")
        used_layer = layer.copy()
        
        used_layer.build(input_size)
        
        if hasattr(used_layer, "output_size"):
            # so far, only the Dense layer has an `output_size` attribute
            output_size = used_layer.output_size
        else:
            output_size = input_size
        
        # setting the name of the currently added layer
        self._set_layer_name(used_layer)
        
        # actually updating the layers and the associated input/output sizes
        self._layers.append(used_layer)
        self._io_sizes.append((input_size, output_size))
    
    
    def __call__(
            self,
            input_layer:  Layer,
            output_layer: Layer
        ) -> Network:
        """
        Builds the layers of the current Network instance using the `__call__`
        API. For example :
        
        input_layer = InputLayer(input_size=784)
        x = input_layer
        x = DenseLayer(64)(x)
        x = ActivationLayer("relu")(x)
        x = DenseLayer(10)(x)
        x = ActivationLayer("softmax")(x)
        output_layer = x
        
        network = Network()(input_layer, output_layer)
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified arguments
        
        assert issubclass(type(input_layer),  Layer)
        assert issubclass(type(output_layer), Layer)
        
        # ------------------------------------------------------------------ #
        
        layers = Layer.TEMPORARY_NETWORK_LAYERS
        
        # the Network instance needs at least an input and an output layer
        assert len(layers) >= 2
        
        input_layer_ID  = input_layer._unique_ID
        output_layer_ID = output_layer._unique_ID
        assert input_layer_ID != output_layer_ID
        
        index_of_input_layer  = None
        index_of_output_layer = None
        
        # retrieving the indices of the specified input and output layers
        # (relative to the `layers` list)
        for layer_index, layer in enumerate(layers):
            layer_ID = layer._unique_ID
            
            if layer_ID == input_layer_ID:
                index_of_input_layer = layer_index
            elif layer_ID == output_layer_ID:
                index_of_output_layer = layer_index
        
        assert index_of_input_layer is not None
        assert index_of_output_layer is not None
        assert index_of_output_layer > index_of_input_layer
        
        # adding the layers of interest to the current Network instance
        for layer_index in range(index_of_input_layer, index_of_output_layer + 1):
            layer = layers[layer_index]
            self.add(layer)
        
        Layer.clear_temporary_network_layers()
        
        return self
    
    
    def _get_total_nb_of_trainable_params(self) -> int:
        """
        Returns the total number of trainable paramaters of the network
        """
        total_nb_of_trainable_params = 0
        
        for layer in self._layers:
            assert issubclass(type(layer), Layer)
            nb_trainable_params = layer.nb_trainable_params
            
            if nb_trainable_params is None:
                raise Exception(f"Network._get_total_nb_of_trainable_params - Please define the `nb_trainable_params` attribute of the \"{layer.__class__.__name__}\" class ! It's still set to the default value (i.e. `None`)")
            
            total_nb_of_trainable_params += nb_trainable_params
        
        return total_nb_of_trainable_params
    
    
    def _get_summary_data(self) -> dict[str, list[str]]:
        """
        Returns the raw data that will be printed in the `Network.summary`
        method. The reason we retrieve the WHOLE summary data before printing
        it is to align the columns of the summary !
        """
        
        # Initializing the summary data with the column titles. Note that the
        # order in which the data is listed MATTERS (when defining this dictionary),
        # since the columns of the printed summary will be in the SAME order !
        summary_data = {
            "layer_names"      : ["Layer name"],
            "layer_types"      : ["Layer type"],
            "input_shapes"     : ["Input shape"],
            "output_shapes"    : ["Output shape"],
            "trainable_params" : ["Trainable parameters"]
        }
        
        assert len(self._layers) == len(self._io_sizes)
        
        for layer, io_size in zip(self._layers, self._io_sizes):
            # -------------------------------------------------------------- #
            
            # retrieving the summary data related to the current layer
            
            layer_name = layer._name
            assert isinstance(layer_name, str)
            
            layer_type = str(layer).replace("Layer", "")
            
            input_size, output_size = io_size
            input_shape  = str((None, input_size))
            output_shape = str((None, output_size))
            
            assert issubclass(type(layer), Layer)
            nb_trainable_params = layer.nb_trainable_params
            if nb_trainable_params is None:
                raise Exception(f"Network._get_summary_data - Please define the `nb_trainable_params` attribute of the \"{layer.__class__.__name__}\" class ! It's still set to the default value (i.e. `None`)")
            nb_trainable_params = "{:,}".format(nb_trainable_params)
            
            # -------------------------------------------------------------- #
            
            # updating the summary data
            
            summary_data["layer_names"].append(layer_name)
            summary_data["layer_types"].append(layer_type)
            summary_data["input_shapes"].append(input_shape)
            summary_data["output_shapes"].append(output_shape)
            summary_data["trainable_params"].append(nb_trainable_params)
            
            # -------------------------------------------------------------- #
        
        return summary_data
    
    
    def summary(
            self,
            *,
            print_summary: bool = True,
            column_separator: str = "|",        # can be multiple characters long
            row_separator: str = "-",           # has to be a single character
            bounding_box: str = "*",            # has to be a single character
            alignment: str = "left",            # = "left" (default), "right" or "center"
            transition_row_style: str = "full", # = "full" (default) or "partial"
            column_spacing: int = 2,
            horizontal_spacing: int = 4,
            vertical_spacing: int = 1,
            offset_spacing: int = 1
        ) -> Union[None, str]:
        """
        Prints the summary of the network's architecture
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified kwargs
        
        assert isinstance(print_summary, bool)
        
        assert isinstance(column_separator, str)
        assert len(column_separator.strip()) >= 1
        column_separator = column_separator.strip()
        
        assert isinstance(row_separator, str)
        assert len(row_separator) == 1
        assert row_separator != " "
        
        assert isinstance(bounding_box, str)
        assert len(bounding_box) == 1
        assert bounding_box != " "
        
        assert isinstance(alignment, str)
        alignment = alignment.strip().lower()
        possible_alignments = ["left", "right", "center"]
        if alignment not in possible_alignments:
            raise ValueError(f"Network.summary - Unrecognized value for the `alignment` kwarg : \"{alignment}\" (possible alignments : {list_to_string(possible_alignments)})")
        
        assert isinstance(transition_row_style, str)
        transition_row_style = transition_row_style.strip().lower()
        possible_transition_row_styles = ["full", "partial"]
        if transition_row_style not in possible_transition_row_styles:
            raise ValueError(f"Network.summary - Unrecognized value for the `transition_row_style` kwarg : \"{transition_row_style}\" (possible styles for the transition row : {list_to_string(possible_transition_row_styles)})")
        
        assert isinstance(column_spacing, int)
        assert column_spacing >= 1
        
        assert isinstance(horizontal_spacing, int)
        assert horizontal_spacing >= 1
        
        assert isinstance(vertical_spacing, int)
        assert vertical_spacing >= 1
        
        assert isinstance(offset_spacing, int)
        assert offset_spacing >= 0
        
        # ------------------------------------------------------------------ #
        
        # checking if the network has got any layers or not
        
        assert len(self._layers) >= 1, "\nNetwork.summary - You can't print the network's summary if it doesn't contain any layers !"
        
        # ------------------------------------------------------------------ #
        
        # getting the raw summary data, and generating all the useful metadata
        # related to the lengths of (the string representations of) the raw
        # summary data, in order to align the columns of the network's summary
        
        summary_data = self._get_summary_data()
        
        nb_aligned_rows = len(self._layers) + 1
        
        for data in summary_data.values():
            assert isinstance(data, list)
            assert len(data) == nb_aligned_rows
            
            for feature in data:
                assert isinstance(feature, str)
                assert len(feature) > 0
                assert feature == feature.strip()
        
        nb_aligned_columns = len(summary_data)
        assert nb_aligned_columns >= 2
        
        # keys   : feature names
        # values : maximum lengths of all the associated features
        maximum_lengths = {}
        for feature_name, data in summary_data.items():
            max_length_of_feature = max([len(feature) for feature in data])
            maximum_lengths[feature_name] = max_length_of_feature
        
        values_of_max_lengths = list(maximum_lengths.values())
        
        horizontal_spacing = " " * horizontal_spacing
        column_spacing     = " " * column_spacing
        
        max_length_of_all_rows = sum(values_of_max_lengths) + nb_aligned_columns * (len(column_separator) + 2 * len(column_spacing)) + len(column_separator)
        
        # ------------------------------------------------------------------ #
        
        # building the row that only contains the `bounding_box` character
        
        top_bottom_border_row  = bounding_box * (max_length_of_all_rows + 2 * (len(bounding_box) + len(horizontal_spacing)))
        
        # ------------------------------------------------------------------ #
        
        # building the blank row with the `bounding_box` character at its ends
        
        blank_row = bounding_box + " " * (max_length_of_all_rows + 2 * len(horizontal_spacing)) + bounding_box
        
        # ------------------------------------------------------------------ #
        
        # building the row containing the title (by default, it will always be
        # centered)
        
        title = "NETWORK SUMMARY"
        
        nb_spaces_title_row = max_length_of_all_rows - len(title)
        title_left_spacing  = " " * (nb_spaces_title_row // 2)
        title_right_spacing = " " * (nb_spaces_title_row - len(title_left_spacing))
        
        title_row = f"{title_left_spacing}{title}{title_right_spacing}"
        full_title_row = f"\n{bounding_box}{horizontal_spacing}{title_row}{horizontal_spacing}{bounding_box}"
        
        # ------------------------------------------------------------------ #
        
        # building the row separating the column titles from the actual summary
        # data (i.e. the "transition row")
        
        transition_row = column_separator
        
        for max_length_of_feature in values_of_max_lengths:
            if transition_row_style == "full":
                transition_row += row_separator * (max_length_of_feature + 2 * len(column_spacing)) + column_separator
            elif transition_row_style == "partial":
                transition_row += column_spacing + row_separator * max_length_of_feature + column_spacing + column_separator
        
        full_transition_row = f"\n{bounding_box}{horizontal_spacing}{transition_row}{horizontal_spacing}{bounding_box}"
        
        # ------------------------------------------------------------------ #
        
        # building the last row, i.e. the row containing the total number of
        # trainable parameters (by default, it will always be centered)
        
        total_nb_of_trainable_params = "{:,}".format(self._get_total_nb_of_trainable_params())
        
        last_printed_row = f"Total number of trainable parameters : {total_nb_of_trainable_params}"
        assert len(last_printed_row) <= max_length_of_all_rows
        
        last_printed_row_spacing = " " * (max_length_of_all_rows - len(last_printed_row))
        left_spacing_last_printed_row  = " " * (len(last_printed_row_spacing) // 2)
        right_spacing_last_printed_row = " " * (len(last_printed_row_spacing) - len(left_spacing_last_printed_row))
        
        last_printed_row = f"{left_spacing_last_printed_row}{last_printed_row}{right_spacing_last_printed_row}"
        full_last_printed_row = f"\n{bounding_box}{horizontal_spacing}{last_printed_row}{horizontal_spacing}{bounding_box}"
        
        # ------------------------------------------------------------------ #
        
        # actually building the string representation of the network's summary
        
        str_summary = f"\n{top_bottom_border_row}"
        str_summary += f"\n{blank_row}" * vertical_spacing
        str_summary += full_title_row
        str_summary += f"\n{blank_row}"
        
        for aligned_row_index in range(nb_aligned_rows):
            # building the current row
            
            current_row = column_separator
            
            for column_index, (feature_name, max_length_of_feature_name) in enumerate(maximum_lengths.items()):
                current_row += column_spacing
                
                feature = summary_data[feature_name][aligned_row_index]
                feature_spacing = " " * (max_length_of_feature_name - len(feature))
                
                if alignment == "left":
                    current_row += f"{feature}{feature_spacing}"
                elif alignment == "right":
                    current_row += f"{feature_spacing}{feature}"
                elif alignment == "center":
                    feature_left_spacing  = " " * (len(feature_spacing) // 2)
                    feature_right_spacing = " " * (len(feature_spacing) - len(feature_left_spacing))
                    current_row += f"{feature_left_spacing}{feature}{feature_right_spacing}"
                
                current_row += f"{column_spacing}{column_separator}"
            
            full_current_row = f"\n{bounding_box}{horizontal_spacing}{current_row}{horizontal_spacing}{bounding_box}"
            
            str_summary += full_current_row
            
            if aligned_row_index == 0:
                # adding the "transition row" right after the row containing
                # the column titles (i.e. the first "aligned row")
                str_summary += full_transition_row
        
        str_summary += f"\n{blank_row}"
        str_summary += full_last_printed_row
        str_summary += f"\n{blank_row}" * vertical_spacing
        str_summary += f"\n{top_bottom_border_row}"
        
        # ------------------------------------------------------------------ #
        
        # adding the offset spacing
        
        offset_spacing = " " * offset_spacing
        str_summary = str_summary.replace("\n", f"\n{offset_spacing}")
        
        # ------------------------------------------------------------------ #
        
        # actually printing/returning the summary
        
        if print_summary:
            print(str_summary)
            return
        
        # for the `Network.__str__` method only
        return str_summary
    
    
    def set_optimizer(
            self,
            optimizer_name: str,
            *,
            learning_rate: float = 0.001,
            verbose: bool = True
        ) -> None:
        """
        Sets the optimizer of (all the layers of) the network to the
        specified optimizer. The optimizer name is case insensitive
        """
        # ------------------------------------------------------------------- #
        
        # checking the validity of the specified arguments
        
        assert isinstance(optimizer_name, str)
        assert len(optimizer_name.strip()) > 0
        optimizer_name = optimizer_name.strip().lower()
        
        assert isinstance(learning_rate, float)
        assert (learning_rate > 0) and (learning_rate < 1)
        
        assert isinstance(verbose, bool)
        
        # ------------------------------------------------------------------- #
        
        for layer in self._layers:
            layer.set_optimizer(optimizer_name, learning_rate=learning_rate)
        
        self.optimizer_name = optimizer_name
        self._learning_rate = learning_rate
        
        if verbose:
            learning_rate_precision = max(2, count_nb_decimals_places(learning_rate))
            str_learning_rate = f"{learning_rate:.{learning_rate_precision}f}"
            print(f"\nThe network's optimizer was successfully set to \"{self.optimizer_name}\" (learning_rate={str_learning_rate})")
    

    def reset_optimizer(self) -> None:
        """
        Resets the optimizer of (all the layers of) the network
        """
        for layer in self._layers:
            layer.reset_optimizer()
    
    
    def set_loss_function(
            self,
            loss_name: str,
            *,
            verbose: bool = True
        ) -> None:
        """
        Sets the loss function of the network. The name of the loss function
        is case insensitive
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified arguments
        
        assert isinstance(loss_name, str)
        loss_name = loss_name.strip().lower()
        if loss_name not in Network.AVAILABLE_LOSSES:
            raise ValueError(f"Network.set_loss_function - Unrecognized loss function name : \"{loss_name}\" (possible loss function names : {list_to_string(list(Network.AVAILABLE_LOSSES))})")
        
        assert isinstance(verbose, bool)
        
        # ------------------------------------------------------------------ #
        
        self.loss_name = loss_name
        self._loss, self._loss_prime = Network.AVAILABLE_LOSSES[self.loss_name]
        
        if verbose:
            print(f"\nThe network's loss function was successfully set to \"{self.loss_name}\"")
    
    
    @staticmethod
    def _validate_data(
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            *,
            input_size_of_network:  Optional[int] = None,
            output_size_of_network: Optional[int] = None,
        ) -> Union[np.ndarray, tuple[np.ndarray]]:
        """
        Checks if the specified (training, validation or test) data is
        valid or not
        
        If `y` is not equal to `None`, `y` can either be a 1D vector of INTEGER
        labels or its one-hot encoded equivalent (in that case it will be a 2D
        matrix). Also, if `y` is not equal to `None`, both `y_categorical` and
        `y_flat` are returned
        """
        # ------------------------------------------------------------------ #
        
        # checking `X`
        
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        
        # checking if `X` has numeric data
        _validate_numpy_dtype(X.dtype)
        
        try:
            check_dtype(X, utils.DEFAULT_DATATYPE)
            used_X = X.copy()
        except:
            used_X = cast(X, utils.DEFAULT_DATATYPE)
        
        nb_features_per_sample = X.shape[1]
        assert nb_features_per_sample >= 2
        
        # ------------------------------------------------------------------ #
        
        # checking `input_size_of_network`
        assert isinstance(input_size_of_network, (type(None), int))
        if (input_size_of_network is not None) and (input_size_of_network != nb_features_per_sample):
            raise ValueError(f"Network._validate_data - The input size of the network (= {input_size_of_network}) doesn't match the number of features per sample in the specified `X` (= {nb_features_per_sample}) !")
        
        # ------------------------------------------------------------------ #
        
        # checking `y` (and `output_size_of_network`)
        
        assert isinstance(y, (type(None), np.ndarray))
        
        if y is not None:
            assert y.ndim in [1, 2]
            
            if y.ndim == 1:
                _validate_label_vector(y)
                y_flat = y.copy()
                y_categorical = vector_to_categorical(y_flat, dtype=utils.DEFAULT_DATATYPE)
            elif y.ndim == 2:
                # checking if `y` has numeric data
                _validate_numpy_dtype(y.dtype)
                
                try:
                    check_dtype(y, utils.DEFAULT_DATATYPE)
                    y_categorical = y.copy()
                except:
                    y_categorical = cast(y, utils.DEFAULT_DATATYPE)
                
                _validate_one_hot_encoded_array(y_categorical)
                y_flat = categorical_to_vector(y_categorical, enable_checks=True)
            
            nb_samples = X.shape[0]
            assert y_categorical.shape[0] == nb_samples
            assert y_flat.size == nb_samples
            
            nb_classes = y_categorical.shape[1]
            assert nb_classes >= 2
            assert np.unique(y_flat).size == nb_classes
            
            # checking `output_size_of_network`
            assert isinstance(output_size_of_network, (type(None), int))
            if (output_size_of_network is not None) and (output_size_of_network != nb_classes):
                raise ValueError(f"Network._validate_data - The output size of the network (= {output_size_of_network}) doesn't match the number of classes in the specified `y` (= {nb_classes}) !")
            
            return used_X, y_categorical, y_flat
        
        return used_X
    
    
    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            nb_epochs: int,
            train_batch_size: int,
            *,
            nb_shuffles_before_each_train_batch_split: int = 10,
            seed_train_batch_splits: Optional[int] = None,
            enable_checks: bool = True,
            validation_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
            val_batch_size: int = 32,
            training_callbacks: Optional[list[Callback]] = None
        ) -> None:
        """
        Trains the network over `nb_epochs` epochs
        
        By design, the network can only be trained *once*
        """
        # ================================================================== #
        
        # basic checks on the specified arguments
        
        # ------------------------------------------------------------------ #
        
        # checking all the args/kwargs (except for `validation_data` and
        # `val_batch_size`)
        
        used_X_train, used_y_train, _ = Network._validate_data(
            X_train,
            y_train,
            input_size_of_network=self._io_sizes[0][0],
            output_size_of_network=self._io_sizes[-1][1]
        )
        nb_train_samples = used_X_train.shape[0]
        cast_nb_train_samples = cast(nb_train_samples, utils.DEFAULT_DATATYPE)
        
        assert isinstance(nb_epochs, int)
        assert nb_epochs > 0
        
        assert isinstance(train_batch_size, int)
        assert train_batch_size > 0
        if train_batch_size > nb_train_samples:
            print(f"\nNetwork.fit - WARNING : train_batch_size > nb_train_samples ({train_batch_size} > {nb_train_samples}), therefore `train_batch_size` was set to `nb_train_samples` (i.e. {nb_train_samples})")
            train_batch_size = nb_train_samples
        
        assert isinstance(nb_shuffles_before_each_train_batch_split, int)
        assert nb_shuffles_before_each_train_batch_split >= 0
        
        if nb_shuffles_before_each_train_batch_split > 0:
            assert isinstance(seed_train_batch_splits, (type(None), int))
            if seed_train_batch_splits is not None:
                assert seed_train_batch_splits >= 0
        
        assert isinstance(enable_checks, bool)
        if not(enable_checks):
            print("\nNetwork.fit - WARNING - The `enable_checks` kwarg is set to `False`. Please make sure you know what you're doing !")
        
        _early_stopping_callback = None
        
        # the check of the `training_callbacks` kwarg isn't optimized, but
        # it's only going to be done once so it doesn't really matter (and it
        # takes less than 0.1 seconds to do anyway)
        assert isinstance(training_callbacks, (type(None), list, tuple))
        if (training_callbacks is not None) and (len(training_callbacks) > 0):
            # checking that the specified callbacks all have a legitimate type
            for callback in training_callbacks:
                assert issubclass(type(callback), Callback)
            
            nb_callbacks = len(training_callbacks)
            
            # checking that the specified callbacks all have a DIFFERENT type
            for callback_index, callback in enumerate(training_callbacks):
                type_current_callback = type(callback)
                for other_callback_index in range(callback_index + 1, nb_callbacks):
                    other_callback = training_callbacks[other_callback_index]
                    if type(other_callback) == type_current_callback:
                        raise ValueError(f"Network.fit - The `training_callbacks` kwarg contains multiple instances of the same \"{type_current_callback.__name__}\" class !")
            
            # checking if the callbacks contain an `EarlyStoppingCallback` instance
            for callback in training_callbacks:
                if isinstance(callback, EarlyStoppingCallback) and (callback.patience < nb_epochs):
                    _early_stopping_callback = callback
                
                # NB : Add an instance check here if you added another
                #      callback class in the "callbacks.py" script
        
        # ------------------------------------------------------------------ #
        
        # checking the `validation_data` and `val_batch_size` kwargs
        
        assert isinstance(validation_data, (type(None), tuple, list))
        
        if validation_data is not None:
            assert len(validation_data) == 2
            X_val, y_val = validation_data
            
            used_X_val, used_y_val, _ = Network._validate_data(
                X_val,
                y_val,
                input_size_of_network=self._io_sizes[0][0],
                output_size_of_network=self._io_sizes[-1][1]
            )
            nb_val_samples = used_X_val.shape[0]
            cast_nb_val_samples = cast(nb_val_samples, utils.DEFAULT_DATATYPE)
            
            # the `val_batch_size` kwarg will not be used if `validation_data`
            # is set to `None`
            assert isinstance(val_batch_size, int)
            assert val_batch_size > 0
            val_batch_size = min(val_batch_size, nb_val_samples)
            
            _has_validation_data = True
        else:
            _has_validation_data = False
            
            if _early_stopping_callback is not None:
                monitored_value = _early_stopping_callback.monitor
                if monitored_value in ["val_loss", "val_accuracy"]:
                    raise ValueError(f"Network.fit - The early stopping callback can't monitor \"{monitored_value}\" if there is no validation data !")
        
        # ================================================================== #
        
        # other checks
        
        if self._is_trained:
            raise Exception("Network.fit - The network has already been trained once !")
        
        if len(self._layers) == 0:
            raise Exception("Network.fit - Please add layers to the network before training it !")
        
        ALLOWED_OUTPUT_ACTIVATIONS = [
            "softmax",
            "log_softmax",
            "sigmoid"
        ]
        
        # checking if the very last layer of the network is a softmax,
        # log-softmax or a sigmoid activation layer
        try:
            last_layer = self._layers[-1]
            assert isinstance(last_layer, ActivationLayer)
            output_activation_name = last_layer.activation_name
            assert output_activation_name in ALLOWED_OUTPUT_ACTIVATIONS
            self._output_activation_is_log_softmax = (output_activation_name == "log_softmax")
        except:
            raise Exception(f"Network.fit - The very last layer of the network must be one of the following activations : {list_to_string(ALLOWED_OUTPUT_ACTIVATIONS)}")
        
        if self.optimizer_name is None:
            raise Exception("Network.fit - Please set an optimizer before training the network !")
        
        if (self._loss is None) or (self._loss_prime is None):
            raise Exception("Network.fit - Please set a loss function before training the network !")
        
        # ================================================================== #
        
        # standardizing the input training and/or validation data (if requested)
        
        if self._standardize_input_data:
            used_X_train = standardize_data(used_X_train)
            if _has_validation_data:
                used_X_val = standardize_data(used_X_val)
        
        # ================================================================== #
        
        # initializing the data for the training loop
        
        # initializing the network's history
        self.history = {
            "epoch"          : [],
            "train_loss"     : [],
            "train_accuracy" : []
        }
        if _has_validation_data:
            self.history["val_loss"]     = []
            self.history["val_accuracy"] = []
        
        nb_train_batches = (nb_train_samples + train_batch_size - 1) // train_batch_size
        
        if nb_shuffles_before_each_train_batch_split == 0:
            train_batches = split_data_into_batches(
                used_X_train,
                train_batch_size,
                labels=used_y_train,
                nb_shuffles=0,
                enable_checks=True
            )
            
            train_batches_data   = train_batches["data"]
            train_batches_labels = train_batches["labels"]
            
            assert len(train_batches_data) == nb_train_batches
        
        if _has_validation_data:
            # Splitting the validation data into batches. Here, you can set
            # the `nb_shuffles` kwarg to any integer you want (or `None`). It's
            # just that, since the order of the data/label batches of the
            # validation set will NOT affect the resulting validation losses and
            # accuracies, you might as well set `nb_shuffles` to zero (to save time)
            val_batches = split_data_into_batches(
                used_X_val,
                val_batch_size,
                labels=used_y_val,
                nb_shuffles=0,
                enable_checks=True
            )
            
            val_batches_data   = val_batches["data"]
            val_batches_labels = val_batches["labels"]
            
            nb_val_batches = len(val_batches_data)
        
        t_end_last_completed_epoch = None
        
        # ================================================================== #
        
        # initializing some variables that'll only be used for display purposes
        
        nb_digits_epoch_index = len(str(nb_epochs))
        epoch_index_format = f"0{nb_digits_epoch_index}d"
        
        nb_digits_train_batch_index = len(str(nb_train_batches))
        train_batch_index_format = f"0{nb_digits_train_batch_index}d"
        
        # Number of times the training batch indices are updated (per epoch).
        # Don't set this value to a value >= 10, otherwise Python's standard
        # output will bug
        nb_train_batch_index_updates = 5
        
        train_batch_index_update_step = nb_train_batches // nb_train_batch_index_updates
        
        # by default
        precision_epoch_history = 4
        assert precision_epoch_history >= 2
        
        # the offset spacing is used to center the prints
        offset_spacing = 5
        
        assert offset_spacing >= 3
        assert offset_spacing % 2 == 1
        offset_spacing = " " * offset_spacing
        
        # `nb_dashes_in_transition` is an empirical value
        if _has_validation_data:
            nb_dashes_in_transition = 61 + 4 * precision_epoch_history + 2 * len(offset_spacing)
        else:
            nb_dashes_in_transition = 29 + 2 * precision_epoch_history + 2 * len(offset_spacing)
        
        transition = "\n# " + "-" * nb_dashes_in_transition + " #"
        
        # by default (it's also an empirical value)
        progress_bar_size = 11 + precision_epoch_history
        
        # ================================================================== #
        
        # variables that will only be used if there is an "early stopping"
        # callback or a `KeyboardInterrupt` exception during the training loop
        
        jupyter_notebook = is_being_run_on_jupyter_notebook()
        
        if not(jupyter_notebook):
            printed_color = Fore.MAGENTA # by default
            reset_color   = Style.RESET_ALL
        else:
            # printing colored text doesn't work in Jupyter notebooks
            printed_color = ""
            reset_color   = ""
        
        colored_message_format = f"\n{offset_spacing}{printed_color}" + "{}" + reset_color
        
        # ================================================================== #
        
        t_beginning_training = perf_counter()
        
        # ================================================================== #
        
        # training loop
        
        _training_loop_was_prematurely_stopped = False
        
        print(transition)
        
        introduction = f"\n{offset_spacing}Starting the training loop ..."
        print(introduction)
        
        seed = seed_train_batch_splits
        
        # `epoch_index` doesn't need to be zero-indexed here
        for epoch_index in range(1, nb_epochs + 1):
            try:
                # for display purposes only
                formatted_epoch_index = format(epoch_index, epoch_index_format)
                epoch_header = f"\n{offset_spacing}epoch {formatted_epoch_index}/{nb_epochs} :"
                
                print(epoch_header)
                
                # NB : Here, `train_batches` is a generator
                if nb_shuffles_before_each_train_batch_split != 0:
                    # splitting the training data into batches
                    train_batches = split_data_into_batches(
                        used_X_train,
                        train_batch_size,
                        labels=used_y_train,
                        is_generator=True,
                        nb_shuffles=nb_shuffles_before_each_train_batch_split,
                        seed=seed,
                        enable_checks=enable_checks
                    )
                else:
                    train_batches = zip(train_batches_data, train_batches_labels)
                
                if seed is not None:
                    # updating the seed in order to make the shuffling of the
                    # training data different at each epoch
                    seed += 1 # the increment value is arbitrary
                
                # re-initializing the network's optimizer
                self.reset_optimizer()
                
                # initializing the training loss and accuracy
                train_loss     = self._zero
                train_accuracy = 0
                
                # initializing the index of the current batch of training data
                train_batch_index = 0
                
                for X_train_batch, y_train_batch in train_batches:
                    # `train_batch_index` doesn't need to be zero-indexed here
                    train_batch_index += 1
                    assert train_batch_index <= nb_train_batches # necessary check
                    
                    # forward propagation
                    train_output = X_train_batch
                    for layer in self._layers:
                        train_output = layer.forward_propagation(
                            train_output,
                            training=True,
                            enable_checks=enable_checks
                        )
                    
                    # updating the (raw) training loss
                    train_loss += np.sum(self._loss(
                        y_train_batch,
                        train_output,
                        y_pred_is_log_softmax_output=self._output_activation_is_log_softmax,
                        enable_checks=enable_checks
                    ))
                    
                    # updating the (raw) training accuracy
                    train_accuracy += accuracy_score(
                        categorical_to_vector(y_train_batch, enable_checks=enable_checks),
                        categorical_to_vector(train_output,  enable_checks=enable_checks),
                        normalize=False,
                        enable_checks=enable_checks
                    )
                    
                    # backward propagation
                    output_gradient = self._loss_prime(
                        y_train_batch,
                        train_output,
                        y_pred_is_log_softmax_output=self._output_activation_is_log_softmax,
                        enable_checks=enable_checks
                    )
                    for layer in reversed(self._layers):
                        output_gradient = layer.backward_propagation(
                            output_gradient,
                            enable_checks=enable_checks
                        )
                        
                        # updating the training loss with the added correction
                        # terms from the L1/L2 regularizations (if there are any)
                        train_loss += layer.loss_leftovers
                    
                    update_progress_bar = (
                        (train_batch_index % train_batch_index_update_step == 0)
                        or
                        (train_batch_index == 1)
                        or
                        (train_batch_index == nb_train_batches)
                    )
                    
                    if update_progress_bar:
                        # displaying the progress bar related to the number of
                        # processed batches (within the current epoch)
                        
                        formatted_batch_index = format(train_batch_index, train_batch_index_format)
                        
                        current_progress_bar = progress_bar(
                            train_batch_index,
                            nb_train_batches,
                            progress_bar_size=progress_bar_size,
                            enable_checks=False
                        )
                        
                        train_batch_progress_row = f"{offset_spacing}{current_progress_bar} batch {formatted_batch_index}/{nb_train_batches}"
                        print(train_batch_progress_row, end="\r")
                
                # necessary check (to see if the correct number of batches were
                # generated during the current epoch)
                assert train_batch_index == nb_train_batches
                
                train_loss /= cast_nb_train_samples
                check_dtype(train_loss, utils.DEFAULT_DATATYPE)
                train_accuracy = float(train_accuracy) / nb_train_samples
                
                # -------------------------------------------------------------- #
                
                # validation step for the current epoch (if requested)
                
                if _has_validation_data:
                    # initializing the validation loss and accuracy
                    val_loss     = self._zero
                    val_accuracy = 0
                    
                    for X_val_batch, y_val_batch in zip(val_batches_data, val_batches_labels):
                        # forward propagation
                        val_output = X_val_batch
                        for layer in self._layers:
                            val_output = layer.forward_propagation(
                                val_output,
                                training=False,
                                enable_checks=enable_checks
                            )
                        
                        # updating the (raw) validation loss
                        val_loss += np.sum(self._loss(
                            y_val_batch,
                            val_output,
                            y_pred_is_log_softmax_output=self._output_activation_is_log_softmax,
                            enable_checks=enable_checks
                        ))
                        
                        # updating the (raw) validation accuracy
                        val_accuracy += accuracy_score(
                            categorical_to_vector(y_val_batch, enable_checks=enable_checks),
                            categorical_to_vector(val_output,  enable_checks=enable_checks),
                            normalize=False,
                            enable_checks=enable_checks
                        )
                    
                    val_loss /= cast_nb_val_samples
                    check_dtype(val_loss, utils.DEFAULT_DATATYPE)
                    val_accuracy = float(val_accuracy) / nb_val_samples
                
                # -------------------------------------------------------------- #
                
                # updating the network's history with the losses and accuracies
                # computed during the current epoch
                
                self.history["epoch"].append(epoch_index)
                self.history["train_loss"].append(train_loss)
                self.history["train_accuracy"].append(train_accuracy)
                
                if _has_validation_data:
                    self.history["val_loss"].append(val_loss)
                    self.history["val_accuracy"].append(val_accuracy)
                
                # -------------------------------------------------------------- #
                
                epoch_history = offset_spacing
                if _has_validation_data:
                    epoch_history += f"train_loss={train_loss:.{precision_epoch_history}f} - val_loss={val_loss:.{precision_epoch_history}f} - train_accuracy={train_accuracy:.{precision_epoch_history}f} - val_accuracy={val_accuracy:.{precision_epoch_history}f}"
                else:
                    epoch_history += f"train_loss={train_loss:.{precision_epoch_history}f} - train_accuracy={train_accuracy:.{precision_epoch_history}f}"
                
                print(epoch_history)
                
                # -------------------------------------------------------------- #
                
                # Checking if there is an early stopping callback (if requested)
                
                if (_early_stopping_callback is not None) and (epoch_index != nb_epochs):
                    prematurely_stop_training_loop = _early_stopping_callback.callback(
                        self.history,
                        enable_checks=enable_checks
                    )
                    
                    if prematurely_stop_training_loop:
                        _training_loop_was_prematurely_stopped = True
                        
                        callback_message = colored_message_format.format(f"{str(_early_stopping_callback)} :")
                        callback_message += colored_message_format.format(f"Prematurely stopping the training loop after epoch n°{epoch_index}")
                        
                        # Enables the ability to print colored text in the standard
                        # output (of Python consoles/terminals). This method is
                        # imported from the `colorama` module
                        init()
                        
                        print(callback_message)
                        
                        # updating the actual number of completed epochs
                        nb_epochs = epoch_index
                        
                        break
                
                # -------------------------------------------------------------- #
                
                t_end_last_completed_epoch = perf_counter()
            
            except KeyboardInterrupt:
                _training_loop_was_prematurely_stopped = True
                
                # Removing the history values that were added during the
                # very last epoch (it's very unlikely, but it's still worth
                # handling)
                for features in self.history.values():
                    if len(features) == epoch_index:
                        features.pop(-1)
                
                keyboard_interrupt_message = "\n" + colored_message_format.format("KeyboardInterrupt :")
                keyboard_interrupt_message += colored_message_format.format(f"The training loop was prematurely stopped during epoch n°{epoch_index}")
                
                # Enables the ability to print colored text in the standard
                # output (of Python consoles/terminals). This method is
                # imported from the `colorama` module
                init()
                
                print(keyboard_interrupt_message)
                
                # updating the actual number of completed epochs
                nb_epochs = max(1, epoch_index - 1)
                
                break
        
        # ================================================================== #
        
        if t_end_last_completed_epoch is not None:
            t_end_training = t_end_last_completed_epoch
        else:
            # in this case, it means that the training loop was interrupted
            # during the very first epoch
            t_end_training = t_beginning_training
        
        duration_training = t_end_training - t_beginning_training
        
        # ================================================================== #
        
        # termination of the training loop
        
        average_epoch_duration = duration_training / nb_epochs

        # NB : this is an approximation of the average processing time of
        # a single *training* batch !
        average_batch_duration = average_epoch_duration / nb_train_batches
        average_batch_duration_in_milliseconds = 1000 * average_batch_duration
        
        space_or_newline = " " * int(_has_validation_data) + f"\n{offset_spacing}" * int(not(_has_validation_data))
        
        conclusion = ""
        if not(_training_loop_was_prematurely_stopped):
            conclusion += f"\n{offset_spacing}Training complete !\n"
        time_precision = 1 # by default
        formatted_duration_training = format_runtime(duration_training, decimal_precision=time_precision)
        conclusion += f"\n{offset_spacing}Done in {formatted_duration_training}{space_or_newline}({average_epoch_duration:.{time_precision}f} s/epoch, {average_batch_duration_in_milliseconds:.{time_precision}f} ms/batch)"
        print(conclusion)
        
        print(transition)
        
        self._is_trained = True
    
    
    def _check_if_trained(self) -> None:
        """
        Checks if the network has already been trained or not (using the
        `Network.fit` method)
        """
        if not(self._is_trained):
            raise Exception("The network hasn't been trained yet !")
    
    
    def plot_history(
            self,
            *,
            save_plot_to_disk: bool = False,
            saved_image_name: str = "network_history"
        ) -> None:
        """
        Plots the evolution of the losses and the accuracies of the network
        during the (last) training phase
        
        If you requested that the plot should be saved to the disk, then, by
        default, it will be saved in the `saved_plots` folder (as a PNG image)
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified arguments
        
        assert isinstance(save_plot_to_disk, bool)
        
        if save_plot_to_disk:
            # the `saved_image_name` kwarg will not be used if
            # `save_plot_to_disk` is set to `False`
            
            assert isinstance(saved_image_name, str)
            assert len(saved_image_name.strip()) > 0
            saved_image_name = saved_image_name.strip()
            saved_image_name = " ".join(saved_image_name.split())
            
            if os.path.sep in saved_image_name:
                raise ValueError("Network.plot_history - Please don't add sub-folder info to the specified `saved_image_name` kwarg ! Just type in the basename of the saved plot")
        
        # ------------------------------------------------------------------ #
        
        # checking if the network has already been trained or not
        
        self._check_if_trained()
        
        # ------------------------------------------------------------------ #
        
        # initialization
        
        if ("val_loss" in self.history) and ("val_accuracy" in self.history):
            _has_validation_data = True
        else:
            _has_validation_data = False
        
        epochs           = self.history["epoch"]
        train_losses     = self.history["train_loss"]
        train_accuracies = self.history["train_accuracy"]
        
        if _has_validation_data:
            val_losses     = self.history["val_loss"]
            val_accuracies = self.history["val_accuracy"]
        
        nb_epochs = len(epochs)
        
        if nb_epochs <= 1:
            if nb_epochs == 0:
                print("\nNetwork.plot_history - WARNING : You cannot plot the network's history if you interrupted the training loop during epoch n°1 !")
            elif nb_epochs == 1:
                print("\nNetwork.plot_history - WARNING : You cannot plot the network's history if you only trained it over 1 epoch !")
            return
        
        # for example
        color_of_train_data = "dodgerblue"
        if _has_validation_data:
            color_of_val_data = "orange"
        
        # ------------------------------------------------------------------ #
        
        # generating the plot
        
        _, ax = plt.subplots(1, 2, figsize=(16, 8))
        plt.suptitle(f"\nHistory of the network's training phase (over {nb_epochs} epochs)", fontsize=15)
        
        # losses subplot
        ax[0].plot(epochs, train_losses, color=color_of_train_data, label="train_loss")
        if _has_validation_data:
            ax[0].plot(epochs, val_losses, color=color_of_val_data, label="val_loss")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True)) # force integer ticks on the x-axis (i.e. the "epoch" axis)
        if _has_validation_data:
            max_loss_value = max(np.max(train_losses), np.max(val_losses))
        else:
            max_loss_value = np.max(train_losses)
        ax[0].set_ylim([0, 1.05 * max_loss_value])
        if _has_validation_data:
            ax[0].legend()
            ax[0].set_title("Losses for the \"train\" and \"val\" datasets")
        else:
            ax[0].set_title("Losses for the \"train\" dataset")
        
        # accuracies subplot
        ax[1].plot(epochs, train_accuracies, color=color_of_train_data, label="train_accuracy")
        if _has_validation_data:
            ax[1].plot(epochs, val_accuracies, color=color_of_val_data, label="val_accuracy")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True)) # force integer ticks on the x-axis (i.e. the "epoch" axis)
        ax[1].set_ylim([0, 1])
        if _has_validation_data:
            ax[1].legend()
            ax[1].set_title("Accuracies for the \"train\" and \"val\" datasets")
        else:
            ax[1].set_title("Accuracies for the \"train\" dataset")
        
        plt.subplots_adjust(top=0.85)
        
        # ------------------------------------------------------------------ #
        
        # saving the plot (if requested)
        
        if save_plot_to_disk:
            # absolute path of the folder containing all the saved plots
            saved_plots_full_folder_name = os.path.join(
                os.getcwd(),
                Network.DEFAULT_SAVED_PLOTS_FOLDER_NAME
            )

            # creating the folder containing the saved plots (if it doesn't exist)
            if not(os.path.exists(saved_plots_full_folder_name)):
                os.mkdir(saved_plots_full_folder_name)
            
            # the PNG extension was mainly chosen as the default extension
            # because it is supported by all the Matplotlib backends
            default_extension = ".png"
            
            # getting the corrected (full) basename of the saved plot
            root_of_saved_image_name = os.path.splitext(saved_image_name)[0].replace(".", "_")
            saved_image_full_name = root_of_saved_image_name + default_extension
            
            # getting the absolute path of the saved plot
            saved_image_path = os.path.join(
                saved_plots_full_folder_name,
                saved_image_full_name
            )
            
            # if a (saved) plot with exactly the same name already exists, we're
            # deleting it
            if os.path.exists(saved_image_path):
                os.remove(saved_image_path)
            
            t_beginning_image_saving = perf_counter()
            
            # actually saving the plot to the disk
            plt.savefig(
                saved_image_path,
                dpi=300, # by default (high resolution)
                format=default_extension[1 : ]
            )
            
            t_end_image_saving = perf_counter()
            duration_image_saving = t_end_image_saving - t_beginning_image_saving
            print(f"\nThe plot was successfully saved to the location \"{saved_image_path}\". Done in {duration_image_saving:.3f} seconds")
        
        # ------------------------------------------------------------------ #
        
        # actually displaying the plot
        plt.show()
    
    
    def predict(
            self,
            X_test: np.ndarray,
            *,
            test_batch_size: int = 32,
            return_logits: bool = True
        ) -> np.ndarray:
        """
        Returns the network's raw prediction (i.e. the logits) for a given
        test input. If the `return_logits` kwarg is set to `False`, then
        the integer predictions will be returned
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified arguments
        
        used_X_test = Network._validate_data(
            X_test,
            input_size_of_network=self._io_sizes[0][0]
        )
        nb_test_samples = used_X_test.shape[0]
        
        assert isinstance(test_batch_size, int)
        assert test_batch_size > 0
        test_batch_size = min(test_batch_size, nb_test_samples)
        
        assert isinstance(return_logits, bool)
        
        # ------------------------------------------------------------------ #
        
        # checking if the network has already been trained or not
        
        self._check_if_trained()
        
        # ------------------------------------------------------------------ #
        
        # potentially normalizing/standardizing the input test data
        
        if self._standardize_input_data:
            used_X_test = standardize_data(used_X_test)
        
        # ------------------------------------------------------------------ #
        
        # Splitting the test data into batches. Here, the `nb_shuffles` kwarg
        # has to be set to zero, otherwise, to compute the test loss and accuracy
        # (in the `Network.evaluate` method), we would need the shuffled labels
        # too. Yet, by design, this function doesn't take in any labels as inputs.
        # Just like for the validation set, while it is true that the order of the
        # data/label batches of the test set will NOT affect the resulting
        # test losses and accuracies, this assumes that the generated test
        # batches contain the information of the label batches as well, which
        # is NOT the case here (by design) !
        # NB : Here, `test_batches` is a generator
        test_batches = split_data_into_batches(
            used_X_test,
            test_batch_size,
            is_generator=True,
            nb_shuffles=0,
            enable_checks=True
        )
        
        test_outputs = []
        
        for X_test_batch in test_batches:
            # forward propagation
            test_output = X_test_batch
            for layer in self._layers:
                test_output = layer.forward_propagation(
                    test_output,
                    training=False,
                    enable_checks=True
                )
            
            test_outputs.append(test_output)
        
        # raw prediction (i.e. the logits)
        y_pred = np.array(np.vstack(tuple(test_outputs)), dtype=used_X_test.dtype)
        assert y_pred.ndim == 2
        
        if not(return_logits):
            # in this case, the 1D vector of INTEGER labels is returned
            y_pred = categorical_to_vector(y_pred, enable_checks=True)
        
        return y_pred
    
    
    def evaluate(
            self,
            X_test: np.ndarray,
            y_test: np.ndarray,
            *,
            top_N_accuracy: int = 2,
            test_batch_size: int = 32
        ) -> tuple[Union[float, np.ndarray]]:
        """
        Computes the network's prediction of `X_test` (i.e. `y_pred`), and
        returns the accuracy score, the "top-N accuracy score", the test
        loss, the mean confidence levels (of the correct and false predictions)
        and the raw confusion matrix of the network
        
        Here, `y_test` can either be a 1D vector of INTEGER labels or its
        one-hot encoded equivalent (in that case it will be a 2D matrix)
        
        The "top-N accuracy" of `y_test` and `y_pred` is defined as the
        proportion of the classes in `y_test` that lie within the `N` most
        probable classes of each prediction of `y_pred` (here, `N` is actually
        the `top_N_accuracy` kwarg)
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified arguments
        
        used_X_test, y_test_categorical, y_test_flat = Network._validate_data(
            X_test,
            y_test,
            input_size_of_network=self._io_sizes[0][0],
            output_size_of_network=self._io_sizes[-1][1]
        )
        
        nb_test_samples = used_X_test.shape[0]
        cast_nb_test_samples = cast(nb_test_samples, utils.DEFAULT_DATATYPE)
        
        nb_classes = y_test_categorical.shape[1]
        
        assert isinstance(top_N_accuracy, int)
        assert (top_N_accuracy >= 1) and (top_N_accuracy <= nb_classes)
        
        assert isinstance(test_batch_size, int)
        assert test_batch_size > 0
        test_batch_size = min(test_batch_size, nb_test_samples)
        
        # ------------------------------------------------------------------ #
        
        # checking if the network has already been trained or not
        
        self._check_if_trained()
        
        # ------------------------------------------------------------------ #
        
        # getting the raw prediction of the network (i.e. the logits)
        y_pred = self.predict(
            used_X_test,
            test_batch_size=test_batch_size
        )
        
        y_pred_flat = categorical_to_vector(y_pred, enable_checks=True)
        
        # ------------------------------------------------------------------ #
        
        # computing the test loss, the test accuracy and the mean
        # confidence levels
        
        # NB : For the test accuracy, we could simply call :
        #      `100 * accuracy_score(y_test_flat, y_pred_flat)`
        #      but, to simulate the fact that we've only got access to one
        #      test batch at a time, we're computing it batch by batch.
        #      The same goes for the test loss and the mean confidence levels
        
        # initializing the test loss, the test accuracy and the mean
        # confidence levels
        test_loss     = self._zero
        test_accuracy = 0
        mean_confidence_level_correct_predictions = 0.0
        mean_confidence_level_false_predictions   = 0.0
        
        for first_index_of_test_batch in range(0, nb_test_samples, test_batch_size):
            last_index_of_test_batch = first_index_of_test_batch + test_batch_size
            
            y_test_batch = y_test_categorical[first_index_of_test_batch : last_index_of_test_batch, :]
            y_pred_batch = y_pred[first_index_of_test_batch : last_index_of_test_batch, :]
            
            # updating the (raw) test loss
            test_loss += np.sum(self._loss(
                y_test_batch,
                y_pred_batch,
                y_pred_is_log_softmax_output=self._output_activation_is_log_softmax,
                enable_checks=True
            ))
            
            y_test_flat_batch = y_test_flat[first_index_of_test_batch : last_index_of_test_batch]
            y_pred_flat_batch = y_pred_flat[first_index_of_test_batch : last_index_of_test_batch]
            
            # updating the (raw) test accuracy
            test_accuracy += accuracy_score(
                y_test_flat_batch,
                y_pred_flat_batch,
                normalize=False,
                enable_checks=True
            )
            
            if not(self._output_activation_is_log_softmax):
                used_y_pred_batch = y_pred_batch
            else:
                used_y_pred_batch = np.exp(y_pred_batch)
            
            # In order to actually get probability distributions, we're
            # normalizing the partial logits (i.e. `y_pred_batch`), in case
            # we're using a final activation function that isn't softmax (like
            # sigmoid for instance)
            normalized_y_pred_batch = used_y_pred_batch / np.sum(used_y_pred_batch, axis=1, keepdims=True)
            
            for batch_sample_index, (actual_class_index, predicted_class_index) in enumerate(zip(y_test_flat_batch, y_pred_flat_batch)):
                current_confidence_level = float(normalized_y_pred_batch[batch_sample_index, predicted_class_index])
                
                # updating the mean confidence levels
                if actual_class_index == predicted_class_index:
                    mean_confidence_level_correct_predictions += current_confidence_level
                else:
                    mean_confidence_level_false_predictions += current_confidence_level
        
        # NB : At this stage, `test_accuracy` is the number of correct predictions
        assert isinstance(test_accuracy, int)
        
        if test_accuracy > 0:
            mean_confidence_level_correct_predictions = 100 * mean_confidence_level_correct_predictions / test_accuracy
        if test_accuracy < nb_test_samples:
            # `nb_test_samples - test_accuracy` is the number of false predictions
            mean_confidence_level_false_predictions = 100 * mean_confidence_level_false_predictions / (nb_test_samples - test_accuracy)
        assert (mean_confidence_level_correct_predictions >= 0) and (mean_confidence_level_correct_predictions <= 100)
        assert (mean_confidence_level_false_predictions >= 0) and (mean_confidence_level_false_predictions <= 100)
        
        test_loss /= cast_nb_test_samples
        check_dtype(test_loss, utils.DEFAULT_DATATYPE)
        test_accuracy = float(test_accuracy) / nb_test_samples
        
        # ------------------------------------------------------------------ #
        
        # computing the accuracy scores and the raw confusion matrix

        forced_nb_digits = 4
        
        acc_score = round(100 * test_accuracy, forced_nb_digits)
        conf_matrix = confusion_matrix(y_test_flat, y_pred_flat)
        
        # checking that the accuracy score computed from the confusion matrix
        # is the same as the one computed by the `accuracy_score` function
        acc_score_from_conf_matrix = round(100 * float(np.sum(np.diag(conf_matrix))) / y_test_flat.size, forced_nb_digits)
        assert np.allclose(acc_score_from_conf_matrix, acc_score)
        
        if top_N_accuracy == 1:
            top_N_acc_score = acc_score
        else:
            # each row of `top_N_integer_predictions` will contain the
            # `top_N_accuracy` most probable predicted classes (in descending
            # order of probability)
            top_N_integer_predictions = np.fliplr(np.argsort(y_pred, axis=1))[:, 0 : top_N_accuracy]

            # this variable is only defined to help fix a weird bug where all the
            # the probabilities (of a specific prediction of `y_pred`) are equal to
            # `1.0 / nb_classes` (it's extremely rare, but possible)
            prediction_with_equal_probabilities = np.full((nb_classes, ), 1.0 / nb_classes, dtype=utils.DEFAULT_DATATYPE)

            # initializing the (raw) top-N accuracy score
            top_N_acc_score = 0
            
            for test_prediction, test_label, top_N_predictions in zip(y_pred, y_test_flat, top_N_integer_predictions):
                # this is only done to prevent a weird bug where all the probabilities
                # (of a specific prediction of `y_pred`) are equal to `1.0 / nb_classes`
                # (it's extremely rare, but possible)
                if np.allclose(
                    test_prediction,
                    prediction_with_equal_probabilities,
                    rtol=0,
                    atol=utils.DTYPE_RESOLUTION
                ):
                    default_test_label = 0 # by design
                    if test_label == default_test_label:
                        top_N_acc_score += 1
                    continue
                
                # by definition of the "top-N accuracy"
                if test_label in top_N_predictions:
                    top_N_acc_score += 1
            
            top_N_acc_score = round(100 * float(top_N_acc_score) / nb_test_samples, forced_nb_digits)
        
        # unit test : we should always have `top_N_acc_score >= acc_score` !
        if top_N_acc_score < acc_score:
            precision_accuracy_scores = 2
            raise Exception(f"\n\ntop-{top_N_accuracy}={top_N_acc_score:.{precision_accuracy_scores}f}%, top-1={acc_score:.{precision_accuracy_scores}f}% : We should have top-{top_N_accuracy} >= top-1 !\n")
        
        # ------------------------------------------------------------------ #
        
        test_results = (
            acc_score,
            top_N_acc_score,
            test_loss,
            mean_confidence_level_correct_predictions,
            mean_confidence_level_false_predictions,
            conf_matrix
        )
        
        return test_results
    
    
    def display_some_predictions(
            self,
            X_test: np.ndarray,
            y_test: np.ndarray,
            *,
            selected_classes: Union[str, list[int]] = "all",
            dict_of_real_class_names: Optional[dict[int, str]] = None,
            image_shape: Optional[tuple[int]] = None,
            seed: Optional[int] = None
        ) -> None:
        """
        Displays some of the network's predictions (of random test samples).
        This method was primarily created for debugging purposes
        
        NB : This function only works if the rows of `X_test` are flattened
             images
        
        Here, `y_test` can either be a 1D vector of INTEGER labels or its
        one-hot encoded equivalent (in that case it will be a 2D matrix)
        
        The kwarg `selected_classes` can either be :
            - the string "all", if you're working with all the classes (default)
            - a list/tuple/1D-array containing the specific class indices you're
              working with (e.g. [2, 4, 7])
        
        The kwarg `dict_of_real_class_names`, if not set to `None`, is a
        dictionary with :
            - as its keys   : all the selected class indices (as integers)
            - as its values : the REAL names of the associated classes (as strings)
        For instance, if you set `selected_classes` to `[2, 4, 7]`, then you
        could, for instance, set `dict_of_real_class_names` to the following
        dictionary :
        dict_of_real_class_names = {
            2 : "TWO",
            4 : "FOUR",
            7 : "SEVEN"
        }
        By default, if `dict_of_real_class_names` is set to `None`, then the
        class names will simply be the string representations of the (selected)
        class indices
        
        If `image_shape` is `None`, then it's automatically assumed that
        the images are square-shaped, i.e. it is assumed that their shape
        is either NxN, NxNx1 or NxNx3
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified arguments
        
        used_X_test, y_test_categorical, y_test_flat = Network._validate_data(
            X_test,
            y_test,
            input_size_of_network=self._io_sizes[0][0],
            output_size_of_network=self._io_sizes[-1][1]
        )
        
        nb_classes = y_test_categorical.shape[1]
        
        _, class_names = _validate_selected_classes(
            selected_classes,
            nb_classes,
            dict_of_real_class_names=dict_of_real_class_names
        )
        
        # checking the validity of the `image_shape` kwarg
        assert isinstance(image_shape, (type(None), tuple, list))
        nb_pixels_per_image = used_X_test.shape[1]
        if image_shape is not None:
            assert len(image_shape) in [2, 3]
            
            assert isinstance(image_shape[0], (int, np.int32, np.int64))
            assert image_shape[0] >= 2
            assert isinstance(image_shape[1], (int, np.int32, np.int64))
            assert image_shape[1] >= 2
            
            if len(image_shape) == 2:
                default_image_shape = image_shape
            elif len(image_shape) == 3:
                assert isinstance(image_shape[2], (int, np.int32, np.int64))
                assert image_shape[2] in [1, 3]
                
                if image_shape[2] == 1:
                    default_image_shape = image_shape[ : 2]
                elif image_shape[2] == 3:
                    default_image_shape = image_shape
            
            # NB : `np.prod(image_shape)` is equal to `np.prod(default_image_shape)`
            assert np.prod(image_shape) == nb_pixels_per_image
        else:
            # here we're assuming that the images are square-shaped, i.e. we're
            # assuming that their shape is either NxN, NxNx1 or NxNx3
            
            sidelength_of_each_image_if_2D = int(round(np.sqrt(nb_pixels_per_image)))
            images_are_2D = (sidelength_of_each_image_if_2D**2 == nb_pixels_per_image)
            
            if images_are_2D:
                # in this case, the images either have a shape of NxN or NxNx1
                default_image_shape = (sidelength_of_each_image_if_2D, sidelength_of_each_image_if_2D)
            else:
                # in this case, the images should have a shape of NxNx3
                
                potential_error_message = "Network.display_some_predictions - The `image_shape` kwarg was set to `None`, but the images in the data aren't square-shaped ! Please set the `image_shape` kwarg manually"
                
                assert nb_pixels_per_image % 3 == 0, potential_error_message
                sidelength_of_each_image_if_3D = int(round(np.sqrt(nb_pixels_per_image // 3)))
                images_are_3D = (3 * sidelength_of_each_image_if_3D**2 == nb_pixels_per_image)
                
                if images_are_3D:
                    default_image_shape = (sidelength_of_each_image_if_3D, sidelength_of_each_image_if_3D, 3)
                else:
                    raise ValueError(potential_error_message)
        
        assert isinstance(seed, (type(None), int))
        if seed is not None:
            assert seed >= 0
        
        # ------------------------------------------------------------------ #
        
        # checking if the network has already been trained or not
        
        self._check_if_trained()
        
        # ------------------------------------------------------------------ #
        
        # doing the predictions of the random test samples
        
        nb_rows    = 2
        nb_columns = 5
        
        nb_predictions = nb_rows * nb_columns
        nb_test_samples = used_X_test.shape[0]

        if nb_predictions >= nb_classes:
            random_test_samples, _, random_test_labels, _ = train_test_split(
                used_X_test,
                y_test_flat,
                train_size=nb_predictions,
                stratify=True, # keeping the same class distribution as `y_test_flat`
                random_state=seed
            )
        else:
            np.random.seed(seed)
            random_test_indices = np.random.choice(
                np.arange(nb_test_samples),
                size=(nb_predictions, ),
                replace=False
            )
            np.random.seed(None) # resetting the seed

            random_test_samples = used_X_test[random_test_indices, :]
            random_test_labels  = y_test_flat[random_test_indices]

        # actually computing the logits (i.e. `y_pred`)
        logits = self.predict(random_test_samples)

        # getting the class indices of the "top-2 integer predictions"
        predicted_class_indices = np.fliplr(np.argsort(logits, axis=1))[:, 0 : 2]
        
        # by default
        confidence_level_precision = 0
        
        # NB : This operation doesn't require changing `predicted_class_indices`,
        #      since `exp` is a strictly increasing function
        if self._output_activation_is_log_softmax:
            logits = np.exp(logits)
        
        # In order to actually get probability distributions, we're normalizing
        # the logits, in case we're using a final activation function that isn't
        # softmax (like sigmoid for instance). Note that this normalization
        # doesn't require changing `predicted_class_indices`
        logits /= np.sum(logits, axis=1, keepdims=True)
        
        # ------------------------------------------------------------------ #
        
        # displaying the predictions
        
        _, ax = plt.subplots(nb_rows, nb_columns, figsize=(16, 8))
        plt.suptitle(f"\nPredictions of {nb_predictions}/{nb_test_samples} random test samples (and their confidence level)", fontsize=15)
        
        for image_index in range(nb_predictions):
            predicted_class_index = predicted_class_indices[image_index, 0]
            predicted_class_name = class_names[predicted_class_index]
            confidence_level_first_choice = 100 * logits[image_index, predicted_class_index]
            
            second_choice_predicted_class_index = predicted_class_indices[image_index, 1]
            second_choice_predicted_class_name = class_names[second_choice_predicted_class_index]
            confidence_level_second_choice = 100 * logits[image_index, second_choice_predicted_class_index]
            
            actual_class_index = random_test_labels[image_index]
            actual_class_name = class_names[actual_class_index]
            
            random_test_image = random_test_samples[image_index, :].reshape(default_image_shape)
            
            row_index = image_index // nb_columns
            column_index = image_index % nb_columns
            
            if len(default_image_shape) == 2:
                ax[row_index, column_index].imshow(random_test_image, cmap="gray")
            elif len(default_image_shape) == 3:
                ax[row_index, column_index].imshow(random_test_image)
            
            subplot_title = f"1st prediction : {predicted_class_name} ({confidence_level_first_choice:.{confidence_level_precision}f}%)"
            subplot_title += f"\n2nd prediction : {second_choice_predicted_class_name} ({confidence_level_second_choice:.{confidence_level_precision}f}%)"
            subplot_title += f"\nActual class : {actual_class_name}"
            
            ax[row_index, column_index].set_title(subplot_title)
            ax[row_index, column_index].axis("off")
        
        plt.subplots_adjust(wspace=0.5, top=0.85)
        
        plt.show()
    
    
    def _pickle(self) -> bytes:
        """
        Returns the pickled/serialized version (i.e. a byte representation)
        of the current Network instance
        """
        variables_of_current_network = self.__dict__.copy()
        
        # the `_layers`, `_loss` and `_loss_prime` attributes aren't
        # pickleable/serializable
        variables_of_current_network.pop("_layers")
        variables_of_current_network.pop("_loss")
        variables_of_current_network.pop("_loss_prime")
        
        # building a pickleable/serializable version of the layers of the network
        pickled_layers = []
        for layer in self._layers:
            layer_name = layer.__class__.__name__
            pickled_layer = layer._pickle()
            pickled_layers.append((layer_name, pickled_layer))
        variables_of_current_network["_layers"] = pickled_layers
        
        pickled_network = dumps(variables_of_current_network)
        assert isinstance(pickled_network, bytes)
        assert len(pickled_network) > 0
        
        return pickled_network
    
    
    @classmethod
    def _load(cls, pickled_network: bytes) -> Network:
        """
        Loads (a copy of) the current Network instance, directly from the
        specified `pickled_network` argument. This method is the inverse of
        the `Network._pickle` method
        """
        assert isinstance(pickled_network, bytes)
        assert len(pickled_network) > 0
        
        loaded_network_as_dict = loads(pickled_network)
        assert isinstance(loaded_network_as_dict, dict)
        assert len(loaded_network_as_dict) > 0
        
        # initializing the loaded layer
        loaded_network = cls.__new__(cls)
        
        for variable_name, variable in loaded_network_as_dict.items():
            assert isinstance(variable_name, str)
            setattr(loaded_network, variable_name, variable)
        
        # setting the `_loss` and `_loss_prime` attributes
        loss_name = loaded_network.loss_name
        if loss_name is not None:
            loaded_network.set_loss_function(loss_name, verbose=False)
        else:
            loaded_network._loss = None
            loaded_network._loss_prime = None
        
        # ------------------------------------------------------------------ #
        
        # re-building the layers of the network
        
        actual_layers = []
        
        pickled_layers = loaded_network._layers
        
        for layer_name, pickled_layer in pickled_layers:
            # getting the associated layer class from the list of available
            # layer types
            layer_class = None
            for layer_type in cls.AVAILABLE_LAYER_TYPES:
                if layer_name == layer_type.__name__:
                    layer_class = layer_type
                    break
            assert layer_class is not None
            
            layer = layer_class._load(pickled_layer)
            actual_layers.append(layer)
        
        loaded_network._layers = actual_layers
        
        # ------------------------------------------------------------------ #
        
        # setting the optimizer of (all the layers of) the loaded network
        optimizer_name = loaded_network.optimizer_name
        learning_rate = loaded_network._learning_rate
        if optimizer_name is not None:
            assert learning_rate is not None
            loaded_network.set_optimizer(
                optimizer_name,
                learning_rate=learning_rate,
                verbose=False
            )
        
        loaded_network._unique_ID = generate_unique_ID()
        
        return loaded_network
    
    
    def copy(self) -> Network:
        """
        Returns a copy of the current Network instance
        """
        network_copy = self.__class__._load(self._pickle())
        
        assert type(network_copy) == type(self)
        assert sorted(list(network_copy.__dict__)) == sorted(list(self.__dict__))
        assert network_copy is not self
        
        return network_copy
    
    
    def save(self, filename: str) -> None:
        """
        Saves the current Network to the disk. By default, the network will
        be saved in the `saved_networks` folder, as a Gunzip file (i.e. as a
        ".gz" file). Note that the specified filename must be the BASENAME of
        the path of the network you're saving (with or without the ".gz" extension)
        
        Also, you can save the current Network instance even if you haven't
        trained it !
        """
        # ------------------------------------------------------------------ #
        
        # checking the specified argument
        
        assert isinstance(filename, str)
        assert len(filename.strip()) > 0
        filename = filename.strip()
        filename = " ".join(filename.split())
        
        if os.path.sep in filename:
            raise ValueError("Network.save - Please don't add sub-folder info to the specified `filename` kwarg ! Just type in the basename of the file that will contain the saved network")
        
        # ------------------------------------------------------------------ #

        # absolute path of the folder containing all the saved networks
        saved_networks_full_folder_name = os.path.join(
            os.getcwd(),
            Network.DEFAULT_SAVED_NETWORKS_FOLDER_NAME
        )
        
        # creating the folder containing the saved networks (if it doesn't exist)
        if not(os.path.exists(saved_networks_full_folder_name)):
            os.mkdir(saved_networks_full_folder_name)
        
        # getting the corrected (full) basename of the saved network
        root_of_saved_network_name = os.path.splitext(filename)[0].replace(".", "_")
        saved_network_full_name = root_of_saved_network_name + ".gz"
        
        # getting the absolute path of the saved network
        saved_network_path = os.path.join(
            saved_networks_full_folder_name,
            saved_network_full_name
        )
        
        # if a (saved) network with exactly the same name already exists, we're
        # deleting it
        if os.path.exists(saved_network_path):
            os.remove(saved_network_path)
        
        # ------------------------------------------------------------------ #
        
        t_beginning_network_saving = perf_counter()
        
        pickled_network = self._pickle()
        
        # actually saving the network to the disk
        with gzip_open(saved_network_path, "wb") as COMPRESSED_NETWORK_FILE:
            COMPRESSED_NETWORK_FILE.write(pickled_network)
        
        t_end_network_saving = perf_counter()
        duration_network_saving = t_end_network_saving - t_beginning_network_saving
        print(f"\nThe network was successfully saved to the location \"{saved_network_path}\". Done in {duration_network_saving:.3f} seconds")
    
    
    @classmethod
    def load_network_from_disk(cls, filename: str) -> Network:
        """
        Loads the network saved at the specified location on your disk. Note
        that the specified filename must be the BASENAME of the path of the
        network you're loading (with or without the ".gz" extension)
        """
        # ------------------------------------------------------------------ #
        
        # checking the specified argument
        
        assert isinstance(filename, str)
        assert len(filename.strip()) > 0
        filename = filename.strip()
        filename = " ".join(filename.split())
        
        if os.path.sep in filename:
            raise ValueError("Network.load_network_from_disk - Please don't add sub-folder info to the specified `filename` kwarg ! Just type in the basename of the file that contains the saved network")
        
        # ------------------------------------------------------------------ #
        
        # getting the corrected (full) basename of the saved network
        root_of_saved_network_name = os.path.splitext(filename)[0].replace(".", "_")
        saved_network_full_name = root_of_saved_network_name + ".gz"
        
        # getting the absolute path of the saved network
        saved_network_path = os.path.join(
            os.getcwd(),
            Network.DEFAULT_SAVED_NETWORKS_FOLDER_NAME,
            saved_network_full_name
        )
        
        # checking if the network that is meant to be loaded actually exists
        assert os.path.exists(saved_network_path)
        
        # ------------------------------------------------------------------ #
        
        t_beginning_network_loading = perf_counter()
        
        pickled_network: bytes = b""
        
        # actually loading the network from the disk
        with gzip_open(saved_network_path, "rb") as COMPRESSED_NETWORK_FILE:        
            pickled_network = COMPRESSED_NETWORK_FILE.read()
        
        assert isinstance(pickled_network, bytes)
        assert len(pickled_network) > 0
        
        loaded_network = cls._load(pickled_network)
        
        t_end_network_loading = perf_counter()
        duration_network_loading = t_end_network_loading - t_beginning_network_loading
        print(f"\nThe network was successfully loaded from the location \"{saved_network_path}\". Done in {duration_network_loading:.3f} seconds")
        
        # ------------------------------------------------------------------ #
        
        return loaded_network

