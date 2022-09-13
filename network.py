# -*- coding: utf-8 -*-

"""
Script defining the main network class
"""

import os
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

# used to force integer ticks on the x-axis of a plot
from matplotlib.ticker import MaxNLocator

import utils
from utils import (
    cast,
    check_dtype,
    standardize_data,
    vector_to_categorical,
    categorical_to_vector,
    _validate_label_vector,
    _validate_selected_classes,
    list_to_string,
    clear_currently_printed_row,
    progress_bar
)

from core import (
    split_data_into_batches,
    accuracy_score,
    confusion_matrix
)

from losses import (
    CCE, CCE_prime,
    MSE, MSE_prime
)

from layers import (
    Layer,
    InputLayer,
    DenseLayer,
    ActivationLayer,
    BatchNormLayer,
    DropoutLayer
)


##############################################################################


class Network:
    """
    Network class
    """
    
    # class variable
    AVAILABLE_LAYER_TYPES = (
        InputLayer,
        DenseLayer,
        ActivationLayer,
        BatchNormLayer,
        DropoutLayer
    )
    
    # class variable
    AVAILABLE_LOSSES = {
        "cce" : (CCE, CCE_prime), # CCE = Categorical Cross-Entropy
        "mse" : (MSE, MSE_prime)  # MSE = Mean Squared Error
    }
    
    def __init__(self, standardize_input_data=True):
        assert isinstance(standardize_input_data, bool)
        self.__standardize_input_data = standardize_input_data
        
        self.layers = []
        
        # list containing the input/output sizes of all the layers of the
        # network (it's a list of tuples of 2 integers)
        self._io_sizes = []
        
        self.loss_name = None
        self.loss = None
        self.loss_prime = None
        
        self.history = None
        self._is_trained = False
    
    
    def __str__(self):
        if len(self.layers) == 0:
            return f"{self.__class__.__name__}()"
        
        # using the default summary kwargs (except for `print_summary`, which
        # has to be set to `False` here, in order to return a string, and not
        # a `NoneType`)
        return self.summary(print_summary=False)
    
    
    def __repr__(self):
        return str(self)
    
    
    def add(self, layer):
        """
        Adds a layer to the network
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified layer
        
        assert issubclass(type(layer), Layer)
        
        if not(isinstance(layer, Network.AVAILABLE_LAYER_TYPES)):
            raise TypeError(f"Network.add - Unrecognized layer type : \"{layer.__class__.__name__}\" (available layer types : {list_to_string(Network.AVAILABLE_LAYER_TYPES)})")
        
        # ------------------------------------------------------------------ #
        
        if isinstance(layer, InputLayer):
            assert len(self.layers) == 0, "\nNetwork.add - ERROR - You cannot add an InputLayer if other layers have already been added to the Network !"
            input_size  = layer.input_size
            output_size = input_size
        else:
            assert len(self.layers) >= 1, f"\nNetwork.add - ERROR - Please add an InputLayer to the network before adding a \"{layer.__class__.__name__}\" !"
            input_size  = self._io_sizes[-1][1] # output size of the previous layer
            
            if isinstance(layer, DenseLayer):
                output_size = layer.output_size
                layer.build(input_size) # actually building the Dense layer
            elif isinstance(layer, (ActivationLayer, BatchNormLayer, DropoutLayer)):
                output_size = input_size
        
        self.layers.append(layer)
        self._io_sizes.append((input_size, output_size))
    
    
    def _get_total_nb_of_trainable_params(self):
        """
        Returns the total number of trainable paramaters of the network
        """
        total_nb_of_trainable_params = 0
        
        for layer in self.layers:
            assert issubclass(type(layer), Layer)
            nb_trainable_params = layer.nb_trainable_params
            
            if nb_trainable_params is None:
                raise Exception(f"Network._get_total_nb_of_trainable_params - Please define the `nb_trainable_params` attribute of the \"{layer.__class__.__name__}\" class !")
            
            total_nb_of_trainable_params += nb_trainable_params
        
        return total_nb_of_trainable_params
    
    
    def _get_summary_data(self):
        """
        Returns the raw data that will be printed in the `Network.summary`
        method. The reason we retrieve the WHOLE summary data before printing
        it is to align the columns of the summary !
        """
        
        # Initializing the summary data with the column titles. Note that the
        # order in which the data is listed MATTERS (when defining this dictionary),
        # since the columns of the printed summary will be in the SAME order !
        summary_data = {
            "layer_types"      : ["Layer"],
            "input_shapes"     : ["Input shape"],
            "output_shapes"    : ["Output shape"],
            "trainable_params" : ["Trainable parameters"]
        }
        
        for layer_index, layer in enumerate(self.layers):
            # -------------------------------------------------------------- #
            
            # retrieving the summary data related to the current layer
            
            layer_type = str(layer).replace("Layer", "")
            
            input_size, output_size = self._io_sizes[layer_index]
            input_shape  = str((None, input_size))
            output_shape = str((None, output_size))
            
            assert issubclass(type(layer), Layer)
            nb_trainable_params = layer.nb_trainable_params
            if nb_trainable_params is None:
                raise Exception(f"Network._get_summary_data - Please define the `nb_trainable_params` attribute of the \"{layer.__class__.__name__}\" class !")
            nb_trainable_params = "{:,}".format(nb_trainable_params)
            
            # -------------------------------------------------------------- #
            
            # updating the summary data
            
            summary_data["layer_types"].append(layer_type)
            summary_data["input_shapes"].append(input_shape)
            summary_data["output_shapes"].append(output_shape)
            summary_data["trainable_params"].append(nb_trainable_params)
            
            # -------------------------------------------------------------- #
        
        return summary_data
    
    
    def summary(
            self,
            print_summary=True,
            column_separator="|",        # can be multiple characters long
            row_separator="-",           # has to be a single character
            bounding_box="*",            # has to be a single character
            alignment="left",            # = "left" (default), "right" or "center"
            transition_row_style="full", # = "full" (default) or "partial"
            column_spacing=2,
            horizontal_spacing=4,
            vertical_spacing=1,
            offset_spacing=1
        ):
        """
        Returns the summary of the network's architecture
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified kwargs
        
        assert isinstance(print_summary, bool)
        
        assert isinstance(column_separator, str)
        assert len(column_separator) >= 1
        assert len(column_separator.strip()) >= 1
        column_separator = column_separator.strip()
        
        assert isinstance(row_separator, str)
        assert len(row_separator) == 1
        assert row_separator != " "
        
        assert isinstance(bounding_box, str)
        assert len(bounding_box) == 1
        assert bounding_box != " "
        
        assert isinstance(alignment, str)
        alignment = alignment.lower()
        possible_alignments = ["left", "right", "center"]
        if alignment not in possible_alignments:
            raise ValueError(f"Network.summary - Unrecognized value for the `alignment` kwarg : \"{alignment}\" (possible alignments : {list_to_string(possible_alignments)})")
        
        assert isinstance(transition_row_style, str)
        transition_row_style = transition_row_style.lower()
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
        
        assert len(self.layers) >= 1, "\nNetwork.summary - ERROR - You can't print the network's summary if it doesn't contain any layers !"
        
        # ------------------------------------------------------------------ #
        
        # getting the raw summary data, and generating all the useful metadata
        # related to the lengths of (the string representations of) the raw
        # summary data, in order to align the columns of the network's summary
        
        summary_data = self._get_summary_data()
        
        nb_aligned_rows = len(self.layers) + 1
        for data in summary_data.values():
            assert isinstance(data, list)
            assert len(data) == nb_aligned_rows
        
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
        
        # for the `__str__` method only
        return str_summary
    
    
    def set_loss_function(self, loss_name):
        """
        Sets the loss function of the network
        """
        # checking the validity of the specified loss function name
        assert isinstance(loss_name, str)
        loss_name = loss_name.lower()
        if loss_name not in Network.AVAILABLE_LOSSES:
            raise ValueError(f"Network.set_loss_function - Unrecognized loss function name : \"{loss_name}\" (possible loss function names : {list_to_string(list(Network.AVAILABLE_LOSSES.keys()))})")
        
        self.loss_name = loss_name
        self.loss, self.loss_prime = Network.AVAILABLE_LOSSES[self.loss_name]
    
    
    def _validate_data(self, X, y=None):
        """
        Checks if the specified (training, validation or testing) data is
        valid or not
        
        If `y` is not equal to `None`, `y` can either be a 1D vector of INTEGER
        labels or its one-hot encoded equivalent (in that case it will be a 2D
        matrix). Also, if `y` is not equal to `None`, both `y_categorical` and
        `y_flat` are returned
        """
        # ------------------------------------------------------------------ #
        
        # checking `X`
        
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2
        check_dtype(X, utils.DEFAULT_DATATYPE)
        
        nb_features_per_sample = X.shape[1] # = nb_pixels_per_image = 28 * 28 = 784
        assert nb_features_per_sample >= 2
        input_size_of_network = self._io_sizes[0][0]
        assert nb_features_per_sample == input_size_of_network
        
        # ------------------------------------------------------------------ #
        
        # checking `y`
        
        assert isinstance(y, (type(None), np.ndarray))
        
        if y is not None:
            assert len(y.shape) in [1, 2]
            if len(y.shape) == 1:
                _validate_label_vector(y)
                y_flat = y.copy()
                y_categorical = vector_to_categorical(y_flat, utils.DEFAULT_DATATYPE)
            elif len(y.shape) == 2:
                y_categorical = y.copy()
                y_flat = categorical_to_vector(y_categorical)
            
            check_dtype(y_categorical, utils.DEFAULT_DATATYPE)
            
            nb_samples = X.shape[0]
            assert y_categorical.shape[0] == nb_samples
            assert y_flat.size == nb_samples
            
            nb_classes = y_categorical.shape[1]
            assert nb_classes >= 2
            assert np.unique(y_flat).size == nb_classes
            output_size_of_network = self._io_sizes[-1][1]
            assert nb_classes == output_size_of_network
            
            return y_categorical, y_flat
    
    
    def fit(
            self,
            X_train,
            y_train,
            nb_epochs,
            train_batch_size,
            learning_rate,
            nb_shuffles_before_each_train_batch_split=10,
            seed_train_batch_splits=None,
            validation_data=None,
            val_batch_size=32
        ):
        """
        Trains the network on `nb_epochs` epochs
        
        By design, the network cannot be trained more than once
        """
        # ================================================================== #
        
        # basic checks on the specified arguments
        
        # ------------------------------------------------------------------ #
        
        # checking all the args/kwargs (except for `validation_data` and
        # `val_batch_size`)
        
        y_train, _ = self._validate_data(X_train, y=y_train)
        nb_train_samples = X_train.shape[0]
        
        assert isinstance(nb_epochs, int)
        assert nb_epochs > 0
        
        assert isinstance(train_batch_size, int)
        assert train_batch_size > 0
        if train_batch_size > nb_train_samples:
            print(f"\nNetwork.fit - WARNING : train_batch_size > nb_train_samples ({train_batch_size} > {nb_train_samples}), therefore `train_batch_size` was set to `nb_train_samples` (i.e. {nb_train_samples})")
            train_batch_size = nb_train_samples
        
        assert isinstance(learning_rate, float)
        assert learning_rate > 0
        
        assert isinstance(nb_shuffles_before_each_train_batch_split, int)
        assert nb_shuffles_before_each_train_batch_split >= 0
        
        if nb_shuffles_before_each_train_batch_split > 0:
            assert isinstance(seed_train_batch_splits, (type(None), int))
            if isinstance(seed_train_batch_splits, int):
                assert seed_train_batch_splits >= 0
        
        # ------------------------------------------------------------------ #
        
        # checking the `validation_data` and `val_batch_size` kwargs
        
        assert isinstance(validation_data, (type(None), tuple, list))
        
        if validation_data is not None:
            assert len(validation_data) == 2
            X_val, y_val = validation_data
            
            y_val, _ = self._validate_data(X_val, y=y_val)
            nb_val_samples = X_val.shape[0]
            
            # the `val_batch_size` kwarg will not be used if `validation_data`
            # is set to `None`
            assert isinstance(val_batch_size, int)
            assert val_batch_size > 0
            val_batch_size = min(val_batch_size, nb_val_samples)
            
            _has_validation_data = True
        else:
            _has_validation_data = False
        
        # ================================================================== #
        
        # other checks
        
        if self._is_trained:
            raise Exception("Network.fit - The network has already been trained once !")
        
        if len(self.layers) == 0:
            raise Exception("Network.fit - Please add layers to the network before training it !")
        
        # checking if the very last layer of the network is a softmax
        # or a sigmoid activation layer
        try:
            last_layer = self.layers[-1]
            assert isinstance(last_layer, ActivationLayer)
            assert last_layer.activation_name in ["softmax", "sigmoid"]
        except:
            raise Exception("Network.fit - The very last layer of the network must be a softmax or a sigmoid activation layer !")
        
        if (self.loss is None) or (self.loss_prime is None):
            raise Exception("Network.fit - Please set a loss function before training the network !")
        
        # ================================================================== #
        
        t_beginning_training = perf_counter()
        
        # ================================================================== #
        
        # potentially standardizing the input training and/or validation data
        
        if self.__standardize_input_data:
            used_X_train = standardize_data(X_train)
            if _has_validation_data:
                used_X_val = standardize_data(X_val)
        else:
            used_X_train = X_train.copy()
            if _has_validation_data:
                used_X_val = X_val.copy()
        
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
        
        if _has_validation_data:
            val_batches = split_data_into_batches(
                used_X_val,
                val_batch_size,
                labels=y_val,
                nb_shuffles=0
            )
            
            nb_val_batches = len(val_batches["data"])
        
        # for the backward propagation
        reversed_layers = self.layers[::-1]
        
        # ================================================================== #
        
        # initializing some variables that'll be used for display purposes only
        
        nb_digits_epoch_index = len(str(nb_epochs))
        epoch_index_format = f"0{nb_digits_epoch_index}d"
        
        nb_digits_train_batch_index = len(str(nb_train_batches))
        train_batch_index_format = f"0{nb_digits_train_batch_index}d"
        
        # number of times the training batch indices are updated (per epoch)
        nb_train_batch_index_updates = 5
        
        train_batch_index_update_step = nb_train_batches // nb_train_batch_index_updates
        
        # `nb_dashes_in_transition` is an empirical value
        if _has_validation_data:
            nb_dashes_in_transition = 105
        else:
            nb_dashes_in_transition = 61
        nb_dashes_in_transition += 2 * nb_digits_epoch_index
        
        transition = "\n# " + "-" * nb_dashes_in_transition + " #"
        
        # to center the prints
        offset_spacing = " " * 5
        
        # ================================================================== #
        
        # training loop
        
        print(transition)
        
        introduction = f"\n{offset_spacing}Starting the training loop ...\n"
        print(introduction)
        
        seed = seed_train_batch_splits
        
        for epoch_index in range(nb_epochs):
            train_batches = split_data_into_batches(
                used_X_train,
                train_batch_size,
                labels=y_train,
                nb_shuffles=nb_shuffles_before_each_train_batch_split,
                seed=seed
            )
            assert len(train_batches["data"]) == nb_train_batches
            
            if seed is not None:
                # updating the seed in order to make the shuffling of the
                # training data different at each epoch
                seed += 1
            
            # for display purposes only
            formatted_epoch_index = format(epoch_index + 1, epoch_index_format)
            
            train_loss     = cast(0, utils.DEFAULT_DATATYPE)
            train_accuracy = 0
            
            for train_batch_index in range(nb_train_batches):
                X_train_batch = train_batches["data"][train_batch_index]
                y_train_batch = train_batches["labels"][train_batch_index]
                
                # forward propagation
                train_output = X_train_batch
                for layer in self.layers:
                    train_output = layer.forward_propagation(train_output, training=True)
                
                # computing the training loss and accuracy (for display purposes only)
                train_loss += np.sum(self.loss(y_train_batch, train_output))
                train_accuracy += accuracy_score(
                    categorical_to_vector(y_train_batch),
                    categorical_to_vector(train_output),
                    normalize=False
                )
                
                # backward propagation
                output_gradient = self.loss_prime(y_train_batch, train_output)
                for layer in reversed_layers:
                    output_gradient = layer.backward_propagation(output_gradient, learning_rate)
                
                if ((train_batch_index + 1) % train_batch_index_update_step == 0) or ((train_batch_index + 1) in [1, nb_train_batches]):
                    formatted_batch_index = format(train_batch_index + 1, train_batch_index_format)
                    
                    current_progress_bar = progress_bar(
                        train_batch_index + 1,
                        nb_train_batches,
                        progress_bar_size=15 # by default
                    )
                    
                    train_batch_progress_row = f"{offset_spacing}epoch {formatted_epoch_index}/{nb_epochs}  -  {current_progress_bar}  -  batch {formatted_batch_index}/{nb_train_batches}"
                    
                    clear_currently_printed_row()
                    print(train_batch_progress_row, end="\r")
            
            train_loss /= cast(nb_train_samples, utils.DEFAULT_DATATYPE)
            check_dtype(train_loss, utils.DEFAULT_DATATYPE)
            train_accuracy = float(train_accuracy) / nb_train_samples
            
            # -------------------------------------------------------------- #
            
            # validation step for the current epoch
            
            if _has_validation_data:
                val_loss     = cast(0, utils.DEFAULT_DATATYPE)
                val_accuracy = 0
                
                for val_batch_index in range(nb_val_batches):
                    X_val_batch = val_batches["data"][val_batch_index]
                    y_val_batch = val_batches["labels"][val_batch_index]
                    
                    # forward propagation
                    val_output = X_val_batch
                    for layer in self.layers:
                        val_output = layer.forward_propagation(val_output, training=False)
                    
                    # computing the validation loss and accuracy (for display purposes only)
                    val_loss += np.sum(self.loss(y_val_batch, val_output))
                    val_accuracy += accuracy_score(
                        categorical_to_vector(y_val_batch),
                        categorical_to_vector(val_output),
                        normalize=False
                    )
                
                val_loss /= cast(nb_val_samples, utils.DEFAULT_DATATYPE)
                check_dtype(val_loss, utils.DEFAULT_DATATYPE)
                val_accuracy = float(val_accuracy) / nb_val_samples
            
            # -------------------------------------------------------------- #
            
            # updating the network's history with the data of the
            # current epoch
            
            self.history["epoch"].append(epoch_index + 1)
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_accuracy)
            
            if _has_validation_data:
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_accuracy)
            
            # -------------------------------------------------------------- #
            
            precision_epoch_history = 4 # by default
            epoch_history = f"{offset_spacing}epoch {formatted_epoch_index}/{nb_epochs}  -  "
            if _has_validation_data:
                epoch_history += f"train_loss={train_loss:.{precision_epoch_history}f}  -  val_loss={val_loss:.{precision_epoch_history}f}  -  train_accuracy={train_accuracy:.{precision_epoch_history}f}  -  val_accuracy={val_accuracy:.{precision_epoch_history}f}"
            else:
                epoch_history += f"train_loss={train_loss:.{precision_epoch_history}f}  -  train_accuracy={train_accuracy:.{precision_epoch_history}f}"
            
            clear_currently_printed_row()
            print(epoch_history)
        
        # ================================================================== #
        
        # termination of the training and validation phases
        
        t_end_training = perf_counter()
        duration_training = t_end_training - t_beginning_training
        
        average_epoch_duration = duration_training / nb_epochs
        if _has_validation_data:
            average_batch_duration = average_epoch_duration / (nb_train_batches + nb_val_batches)
        else:
            average_batch_duration = average_epoch_duration / nb_train_batches
        average_batch_duration_in_milliseconds = 1000 * average_batch_duration
        
        conclusion = f"\n{offset_spacing}Training complete !\n\n{offset_spacing}Done in {duration_training:.1f} seconds ({average_epoch_duration:.1f} s/epoch, {average_batch_duration_in_milliseconds:.1f} ms/batch)"
        print(conclusion)
        
        print(transition)
        
        self._is_trained = True
    
    
    def _check_if_trained(self):
        """
        Checks if the network has already been trained or not (using the
        `fit` method)
        """
        if not(self._is_trained):
            raise Exception("The network hasn't been trained yet !")
    
    
    def plot_history(
            self,
            save_plot_to_disk=False,
            saved_image_name="network_history"
        ):
        """
        Plots the evolution of the losses and the accuracies of the network
        during the (last) training phase
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified arguments
        
        assert isinstance(save_plot_to_disk, bool)
        
        if save_plot_to_disk:
            # the `saved_image_name` kwarg will not be used if
            # `save_plot_to_disk` is set to `False`
            assert isinstance(saved_image_name, str)
            assert len(saved_image_name) > 0
            assert len(saved_image_name.strip()) > 0
            saved_image_name = saved_image_name.strip()
        
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
        if nb_epochs == 1:
            print("\nNetwork.plot_history - WARNING : You cannot plot the network's history if you only trained it on 1 epoch !")
            return
        
        # for example
        color_of_train_data = "dodgerblue"
        if _has_validation_data:
            color_of_val_data = "orange"
        
        # ------------------------------------------------------------------ #
        
        # generating the plot
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        plt.suptitle(f"\nHistory of the network's training phase (on {nb_epochs} epochs)", fontsize=15)
        
        # losses
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
        
        # accuracies
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
            # creating the folder containing the saved plot (if it doesn't exist)
            DEFAULT_SAVED_IMAGES_FOLDER_NAME = "saved_plots"
            if not(os.path.exists(DEFAULT_SAVED_IMAGES_FOLDER_NAME)):
                os.mkdir(DEFAULT_SAVED_IMAGES_FOLDER_NAME)
            
            if saved_image_name[-4 : ] != ".png": 
                saved_image_full_name = saved_image_name + ".png"
            else:
                saved_image_full_name = saved_image_name
            
            # getting the absolute path of the saved plot
            saved_image_path = os.path.join(
                os.getcwd(),
                DEFAULT_SAVED_IMAGES_FOLDER_NAME,
                saved_image_full_name
            )
            
            t_beginning_image_saving = perf_counter()
            
            # actually saving the plot to your disk
            plt.savefig(
                saved_image_path,
                dpi=300,
                format="png"
            )
            
            t_end_image_saving = perf_counter()
            duration_image_saving = t_end_image_saving - t_beginning_image_saving
            print(f"\nThe plot was successfully saved to the location \"{saved_image_path}\". Done in {duration_image_saving:.3f} seconds")
        
        # ------------------------------------------------------------------ #
        
        # actually displaying the plot
        plt.show()
    
    
    def predict(self, X_test, test_batch_size=32, return_logits=True):
        """
        Returns the network's raw prediction (i.e. the logits) for a given
        test input
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified arguments
        
        self._validate_data(X_test)
        nb_test_samples = X_test.shape[0]
        
        assert isinstance(test_batch_size, int)
        assert test_batch_size > 0
        test_batch_size = min(test_batch_size, nb_test_samples)
        
        assert isinstance(return_logits, bool)
        
        # ------------------------------------------------------------------ #
        
        # checking if the network has already been trained or not
        
        self._check_if_trained()
        
        # ------------------------------------------------------------------ #
        
        # potentially normalizing/standardizing the input testing data
        
        if self.__standardize_input_data:
            used_X_test = standardize_data(X_test)
        else:
            used_X_test = X_test.copy()
        
        # ------------------------------------------------------------------ #
        
        test_batches = split_data_into_batches(
            used_X_test,
            test_batch_size,
            labels=None,
            nb_shuffles=0
        )
        nb_test_batches = len(test_batches["data"])
        
        test_outputs = []
        
        for test_batch_index in range(nb_test_batches):
            X_test_batch = test_batches["data"][test_batch_index]
            
            # forward propagation
            test_output = X_test_batch
            for layer in self.layers:
                test_output = layer.forward_propagation(test_output, training=False)
            
            test_outputs.append(test_output)
        
        # raw prediction (i.e. the logits)
        y_pred = np.array(np.vstack(tuple(test_outputs)), dtype=used_X_test.dtype)
        assert len(y_pred.shape) == 2
        
        if not(return_logits):
            # in this case, the 1D vector of INTEGER labels is returned
            y_pred = categorical_to_vector(y_pred)
        
        return y_pred
    
    
    def evaluate(
            self,
            X_test,
            y_test,
            top_N_accuracy=2,
            test_batch_size=32
        ):
        """
        Computes the network's prediction of `X_test` (i.e. `y_pred`), then
        returns the accuracy score and the raw confusion matrix of `y_test`
        and `y_pred`
        
        Here, `y_test` can either be a 1D vector of INTEGER labels or its
        one-hot encoded equivalent (in that case it will be a 2D matrix)
        
        The "top-N accuracy" of `y_test` and `y_pred` is also returned. It's
        defined as the proportion of the classes in `y_true` that lie within
        the `N` most probable classes of each prediction of `y_pred` (here, `N`
        is actually the `top_N_accuracy` kwarg)
        
        The testing loss is also returned
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified arguments
        
        y_test_categorical, y_test_flat = self._validate_data(X_test, y=y_test)
        
        nb_test_samples = X_test.shape[0]
        nb_classes = np.unique(y_test_flat).size
        
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
            X_test,
            test_batch_size=test_batch_size
        )
        
        y_pred_flat = categorical_to_vector(y_pred)
        
        # ------------------------------------------------------------------ #
        
        # computing the testing loss and accuracy
        
        test_loss     = cast(0, utils.DEFAULT_DATATYPE)
        test_accuracy = 0
        
        for first_index_of_test_batch in range(0, nb_test_samples, test_batch_size):
            last_index_of_test_batch = first_index_of_test_batch + test_batch_size
            
            y_test_batch = y_test_categorical[first_index_of_test_batch : last_index_of_test_batch, :]
            y_pred_batch = y_pred[first_index_of_test_batch : last_index_of_test_batch, :]
            
            test_loss += np.sum(self.loss(y_test_batch, y_pred_batch))
            test_accuracy += accuracy_score(
                y_test_flat[first_index_of_test_batch : last_index_of_test_batch],
                y_pred_flat[first_index_of_test_batch : last_index_of_test_batch],
                normalize=False
            )
        
        test_loss /= cast(nb_test_samples, utils.DEFAULT_DATATYPE)
        check_dtype(test_loss, utils.DEFAULT_DATATYPE)
        test_accuracy = float(test_accuracy) / nb_test_samples
        
        # ------------------------------------------------------------------ #
        
        # computing the accuracy scores and the raw confusion matrix
        
        acc_score = 100 * test_accuracy
        conf_matrix = confusion_matrix(y_test_flat, y_pred_flat)
        
        # checking that the accuracy score computed from the confusion matrix
        # is the same as the one computed by the `accuracy_score` function
        acc_score_from_conf_matrix = 100 * float(np.sum(np.diag(conf_matrix))) / y_test_flat.size
        assert np.allclose(acc_score_from_conf_matrix, acc_score)
        
        if top_N_accuracy == 1:
            top_N_acc_score = acc_score
        else:
            # each row of `top_N_integer_predictions` will contain the
            # `top_N_accuracy` most probable predicted classes (in descending
            # order of probability)
            top_N_integer_predictions = np.fliplr(np.argsort(y_pred, axis=1))[:, 0 : top_N_accuracy]
            
            top_N_acc_score = 0
            
            for test_label, top_N_predictions in zip(y_test_flat, top_N_integer_predictions):
                # by definition of the "top-N accuracy"
                if test_label in top_N_predictions:
                    top_N_acc_score += 1
            
            nb_test_samples = y_test_flat.size
            top_N_acc_score = 100 * float(top_N_acc_score) / nb_test_samples
        
        assert top_N_acc_score >= acc_score
        
        return acc_score, top_N_acc_score, test_loss, conf_matrix
    
    
    def display_some_predictions(
            self,
            X_test,
            y_test,
            selected_classes="all",
            seed=None
        ):
        """
        Displays the predictions of random test samples
        
        Here, `y_test` can either be a 1D vector of INTEGER labels or its
        one-hot encoded equivalent (in that case it will be a 2D matrix)
        
        The kwarg `selected_classes` can either be :
            - the string "all" (if you're working with all the digits ranging
              from 0 to 9)
            - a list/tuple/1D-array containing the specific digits you're
              working with (e.g. [2, 4, 7])
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified arguments
        
        _, y_test_flat = self._validate_data(X_test, y=y_test)
        
        selected_classes = _validate_selected_classes(selected_classes)
        if isinstance(selected_classes, str):
            # here, `selected_classes` is equal to the string "all"
            nb_classes = np.unique(y_test_flat).size
            classes = np.arange(nb_classes)
        else:
            classes = selected_classes
        
        assert isinstance(seed, (type(None), int))
        if isinstance(seed, int):
            assert seed >= 0
        
        # ------------------------------------------------------------------ #
        
        # checking if the network has already been trained or not
        
        self._check_if_trained()
        
        # ------------------------------------------------------------------ #
        
        # doing the predictions of the random test samples
        
        nb_rows    = 2
        nb_columns = 5
        
        nb_predictions = nb_rows * nb_columns
        nb_test_samples = X_test.shape[0]
        
        np.random.seed(seed)
        random_test_indices = np.random.choice(np.arange(nb_test_samples), size=(nb_predictions, ), replace=False)
        np.random.seed(None) # resetting the seed
        
        random_test_samples = X_test[random_test_indices, :]
        logits = self.predict(random_test_samples) # = y_pred
        
        # getting the "top-2 integer predictions"
        predicted_digit_values = np.fliplr(np.argsort(logits, axis=1))[:, 0 : 2]
        
        # by default
        confidence_level_precision = 0
        
        # in order to actually get probability distributions, we're normalizing
        # the logits (in case we're using a final activation function that isn't
        # softmax, like sigmoid for instance)
        logits /= np.sum(logits, axis=1, keepdims=True)
        
        # here we're assuming that the images are square-shaped
        nb_pixels_per_image = X_test.shape[1]
        sidelength_of_each_image = int(round(np.sqrt(nb_pixels_per_image)))
        assert sidelength_of_each_image**2 == nb_pixels_per_image
        default_image_shape = (sidelength_of_each_image, sidelength_of_each_image)
        
        # ------------------------------------------------------------------ #
        
        # displaying the predictions
        
        fig, ax = plt.subplots(nb_rows, nb_columns, figsize=(16, 8))
        plt.suptitle(f"\nPredictions of {nb_predictions}/{nb_test_samples} random test samples (and their confidence level)", fontsize=15)
        
        for image_index in range(nb_predictions):
            predicted_digit_value_index = predicted_digit_values[image_index, 0]
            predicted_digit_value = classes[predicted_digit_value_index]
            confidence_level_first_choice = 100 * logits[image_index, predicted_digit_value_index]
            
            second_choice_predicted_digit_index = predicted_digit_values[image_index, 1]
            second_choice_predicted_digit = classes[second_choice_predicted_digit_index]
            confidence_level_second_choice = 100 * logits[image_index, second_choice_predicted_digit_index]
            
            actual_digit_value_index = y_test_flat[random_test_indices[image_index]]
            actual_digit_value = classes[actual_digit_value_index]
            
            random_test_image = random_test_samples[image_index, :].reshape(default_image_shape)
            
            row_index = image_index // nb_columns
            column_index = image_index % nb_columns
            ax[row_index, column_index].imshow(random_test_image, cmap="gray")
            
            subplot_title = f"1st prediction : {predicted_digit_value} ({confidence_level_first_choice:.{confidence_level_precision}f}%)"
            subplot_title += f"\n2nd prediction : {second_choice_predicted_digit} ({confidence_level_second_choice:.{confidence_level_precision}f}%)"
            subplot_title += f"\nActual value : {actual_digit_value}"
            
            ax[row_index, column_index].set_title(subplot_title)
            ax[row_index, column_index].axis("off")
        
        plt.subplots_adjust(wspace=0.5, top=0.85)
        
        plt.show()

