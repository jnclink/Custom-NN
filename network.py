# -*- coding: utf-8 -*-

"""
Main network class
"""

import os
from time import time
import numpy as np
import matplotlib.pyplot as plt

# used to force integer ticks on the x-axis of a plot
from matplotlib.ticker import MaxNLocator

import utils
from utils import (
    cast,
    check_dtype,
    split_data_into_batches,
    categorical_to_vector,
    accuracy_score,
    confusion_matrix
)

from losses import (
    MSE, MSE_prime,
    CCE, CCE_prime
)

from layers import (
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
    AVAILABLE_LOSSES = {
        "mse" : (MSE, MSE_prime), # MSE = Mean Squared Error
        "cce" : (CCE, CCE_prime)  # CCE = Categorical Cross-Entropy
    }
    
    # class variable
    AVAILABLE_LAYER_TYPES = (
        InputLayer,
        DenseLayer,
        ActivationLayer,
        BatchNormLayer,
        DropoutLayer
    )
    
    def __init__(self, normalize_input_data=True):
        self.layers = []
        
        # list containing the input/output sizes of all the layers of the
        # network (it's a list of tuples of integers)
        self._sizes = []
        
        self.loss_name = None
        self.loss = None
        self.loss_prime = None
        
        self.history = None
        
        self.__normalize_input_data = normalize_input_data
    
    
    def __str__(self):
        if len(self.layers) == 0:
            return f"{self.__class__.__name__}()"
        
        # using the default summary kwargs (except for `print_summary`, which
        # has to be set to `False` here, in order to return a string, and not
        # a `NoneType`)
        return self.summary(
            initial_spacing=1,
            column_separator="|",
            row_separator="-",
            bounding_box="*",
            print_summary=False
        )
    
    
    def add(self, layer):
        """
        Adds a layer to the network
        """
        if not(isinstance(layer, Network.AVAILABLE_LAYER_TYPES)):
            raise TypeError(f"Network.add - Unrecognized layer type : \"{layer.__class__.__name__}\"")
        
        if isinstance(layer, InputLayer):
            assert len(self.layers) == 0, "\nNetwork.add - ERROR - You cannot add an InputLayer if other layers have already been added to the Network !"
            input_size  = layer.input_size
            output_size = input_size
        else:
            assert len(self.layers) >= 1, f"\nNetwork.add - ERROR - Please add an InputLayer to the network before adding a \"{layer.__class__.__name__}\" !"
            input_size  = self._sizes[-1][1] # output size of the previous layer
            
            if isinstance(layer, DenseLayer):
                output_size = layer.output_size
                layer.build(input_size) # actually building the Dense layer
            elif isinstance(layer, (ActivationLayer, BatchNormLayer, DropoutLayer)):
                output_size = input_size
        
        self.layers.append(layer)
        self._sizes.append((input_size, output_size))
    
    
    def _get_total_nb_of_trainable_params(self):
        """
        Returns the total number of trainable paramaters of the network
        """
        total_nb_of_trainable_params = 0
        for layer in self.layers:
            total_nb_of_trainable_params += layer.nb_trainable_params
        return total_nb_of_trainable_params
    
    
    def _get_summary_data(self):
        """
        Returns the raw data that will be printed in the `summary` method. The
        reason we collect the entire summary data before printing it is to align
        the columns of the summary
        """
        
        # initializing the summary data (with the column titles)
        summary_data = {
            "layer_types"      : ["Layer"],
            "input_shapes"     : ["Input shape"],
            "output_shapes"    : ["Output shape"],
            "trainable_params" : ["Trainable parameters"]
        }
        
        for layer_index, layer in enumerate(self.layers):
            layer_type = str(layer).replace("Layer", "")
            
            input_size, output_size = self._sizes[layer_index]
            input_shape  = str((None, input_size))
            output_shape = str((None, output_size))
            
            nb_trainable_params = "{:,}".format(layer.nb_trainable_params)
            
            # -------------------------------------------------------------- #
            
            # updating the summary data
            
            summary_data["layer_types"].append(layer_type)
            summary_data["input_shapes"].append(input_shape)
            summary_data["output_shapes"].append(output_shape)
            summary_data["trainable_params"].append(nb_trainable_params)
        
        return summary_data
    
    
    def summary(
            self,
            initial_spacing=1,
            column_separator="|",
            row_separator="-",
            bounding_box="*",
            print_summary=True
        ):
        """
        Returns the summary of the network's architecture
        """
        
        assert len(self.layers) >= 1, "\nNetwork.summary - ERROR - You can't print the newtork's summary if it doesn't contain any layers !"
        
        assert initial_spacing >= 0
        assert len(column_separator) >= 1
        assert len(row_separator) == 1
        assert len(bounding_box) == 1
        
        summary_data = self._get_summary_data()
        nb_aligned_rows = len(self.layers) + 1
        
        layer_types      = summary_data["layer_types"]
        input_shapes     = summary_data["input_shapes"]
        output_shapes    = summary_data["output_shapes"]
        trainable_params = summary_data["trainable_params"]
        
        max_len_layer_types      = np.max([len(layer_type) for layer_type in layer_types])
        max_len_input_shapes     = np.max([len(input_shape) for input_shape in input_shapes])
        max_len_output_shapes    = np.max([len(output_shape) for output_shape in output_shapes])
        max_len_trainable_params = np.max([len(nb_trainable_params) for nb_trainable_params in trainable_params])
        
        max_len_of_all_rows = max_len_layer_types + max_len_input_shapes + max_len_output_shapes + max_len_trainable_params + 3 * (len(column_separator) + 4)
        
        full_row = bounding_box * (max_len_of_all_rows + 10)
        empty_row = bounding_box + " " * (max_len_of_all_rows + 8) + bounding_box
        
        # the title is centered by default
        title = "NETWORK SUMMARY"
        nb_spaces_title_row = max_len_of_all_rows - len(title)
        left_spacing_title = " " * (nb_spaces_title_row // 2)
        right_spacing_title = " " * (nb_spaces_title_row - len(left_spacing_title))
        
        str_summary = f"\n{full_row}\n{empty_row}"
        str_summary += f"\n{bounding_box}    {left_spacing_title}{title}{right_spacing_title}    {bounding_box}"
        str_summary += f"\n{empty_row}"
        
        for aligned_row_index in range(nb_aligned_rows):
            layer_type = layer_types[aligned_row_index]
            input_shape = input_shapes[aligned_row_index]
            output_shape = output_shapes[aligned_row_index]
            nb_trainable_params = trainable_params[aligned_row_index]
            
            layer_type_spacing       = " " * (max_len_layer_types - len(layer_type))
            input_shape_spacing      = " " * (max_len_input_shapes - len(input_shape))
            output_shape_spacing     = " " * (max_len_output_shapes - len(output_shape))
            trainable_params_spacing = " " * (max_len_trainable_params - len(nb_trainable_params))
            
            str_row = f"\n{bounding_box}    {layer_type}{layer_type_spacing}  {column_separator}  {input_shape}{input_shape_spacing}  {column_separator}  {output_shape}{output_shape_spacing}  {column_separator}  {nb_trainable_params}{trainable_params_spacing}    {bounding_box}"
            str_summary += str_row
            
            # row separating the column titles from the actual summary data
            if aligned_row_index == 0:
                transition = row_separator * (max_len_layer_types + 2) + column_separator + row_separator * (max_len_input_shapes + 4) + column_separator + row_separator * (max_len_output_shapes + 4) + column_separator + row_separator * (max_len_trainable_params + 2)
                str_summary += f"\n{bounding_box}    {transition}    {bounding_box}"
        
        str_summary += f"\n{empty_row}"
        
        total_nb_of_trainable_params = "{:,}".format(self._get_total_nb_of_trainable_params())
        last_printed_row = f"Total number of trainable parameters : {total_nb_of_trainable_params}"
        nb_spaces_last_printed_row = " " * (max_len_of_all_rows - len(last_printed_row))
        str_summary += f"\n{bounding_box}    {last_printed_row}{nb_spaces_last_printed_row}    {bounding_box}"
        
        str_summary += f"\n{empty_row}\n{full_row}"
        
        # adding the initial spacing
        initial_spacing = " " * initial_spacing
        str_summary = str_summary.replace("\n", f"\n{initial_spacing}")
        
        if print_summary:
            print(str_summary)
            return
        
        # for the `__str__` method only
        return str_summary
    
    
    def set_loss_function(self, loss_name):
        """
        Sets the loss function of the network
        """
        assert isinstance(loss_name, str)
        loss_name = loss_name.lower()
        if loss_name not in list(Network.AVAILABLE_LOSSES.keys()):
            raise ValueError(f"Network.set_loss_function - Unrecognized loss function name : \"{loss_name}\"")
        
        self.loss_name = loss_name
        self.loss, self.loss_prime = Network.AVAILABLE_LOSSES[self.loss_name]
    
    
    def fit(
            self,
            training_data,
            validation_data,
            nb_epochs,
            learning_rate,
            train_batch_size,
            nb_shuffles_before_train_batch_splits=10,
            seed_train_batch_splits=None,
            val_batch_size=32
        ):
        """
        Trains the network on `nb_epochs` epochs
        """
        
        if len(self.layers) == 0:
            raise Exception("Network.fit - Please add layers to the network before training it !")
        
        if (self.loss is None) or (self.loss_prime is None):
            raise Exception("Network.fit - Please set a loss function before training the network !")
        
        t_beginning_training = time()
        
        # initializing the network's history
        self.history = {
            "epoch"        : [],
            "loss"         : [],
            "val_loss"     : [],
            "accuracy"     : [],
            "val_accuracy" : []
        }
        
        X_train, y_train = training_data
        nb_train_samples = X_train.shape[0]
        
        if train_batch_size > nb_train_samples:
            print(f"\nNetwork.fit - WARNING : train_batch_size > nb_train_samples ({train_batch_size} > {nb_train_samples}), therefore `train_batch_size` was set to `nb_train_samples` (i.e. {nb_train_samples})")
            train_batch_size = nb_train_samples
        
        nb_train_batches = (nb_train_samples + train_batch_size - 1) // train_batch_size
        
        X_val, y_val = validation_data
        nb_val_samples = X_val.shape[0]
        
        val_batches = split_data_into_batches(
            X_val,
            y_val,
            batch_size=min(val_batch_size, nb_val_samples),
            normalize_batches=self.__normalize_input_data,
            nb_shuffles=0
        )
        nb_val_batches = len(val_batches["data"])
        
        # for the backward propagation
        reversed_layers = self.layers[::-1]
        learning_rate = cast(learning_rate, utils.DEFAULT_DATATYPE)
        
        # ================================================================== #
        
        # for display purposes only
        
        nb_digits_epoch_index = len(str(nb_epochs))
        epoch_index_format = f"0{nb_digits_epoch_index}d"
        
        nb_digits_train_batch_index = len(str(nb_train_batches))
        train_batch_index_format = f"0{nb_digits_train_batch_index}d"
        
        nb_train_batch_displays = 5 # number of times the batch indices are displayed (per epoch)
        train_batch_index_step = nb_train_batches // nb_train_batch_displays
        
        nb_dashes_in_transition = 93 + 2 * nb_digits_epoch_index # empirical value
        transition = "\n# " + "-" * nb_dashes_in_transition + " #"
        
        initial_spacing = " " * 5 # to center the prints
        
        # used to clear the currently printed line
        blank_row_with_carriage_return = " " * 150 + "\r"
        
        # ================================================================== #
        
        # training loop
        
        print(transition)
        
        introduction = f"\n{initial_spacing}Starting the training loop ...\n"
        print(introduction)
        
        seed = seed_train_batch_splits
        
        for epoch_index in range(nb_epochs):
            train_batches = split_data_into_batches(
                X_train,
                y_train,
                train_batch_size,
                normalize_batches=self.__normalize_input_data,
                nb_shuffles=nb_shuffles_before_train_batch_splits,
                seed=seed
            )
            assert len(train_batches["data"]) == nb_train_batches
            
            if seed is not None:
                # updating the seed in order to make the shuffling of the
                # training data different at each epoch
                seed += 1
            
            # for display purposes only
            formatted_epoch_index = format(epoch_index + 1, epoch_index_format)
            
            loss     = cast(0.0, utils.DEFAULT_DATATYPE)
            accuracy = 0.0
            
            for train_batch_index in range(nb_train_batches):
                X_train_batch = train_batches["data"][train_batch_index]
                y_train_batch = train_batches["labels"][train_batch_index]
                
                # forward propagation
                train_output = X_train_batch
                for layer in self.layers:
                    train_output = layer.forward_propagation(train_output, training=True)
                
                # computing the loss and accuracy (for display purposes only)
                loss += np.sum(self.loss(y_train_batch, train_output))
                accuracy += accuracy_score(
                    categorical_to_vector(y_train_batch),
                    categorical_to_vector(train_output),
                    normalize=False
                )
                
                # backward propagation
                backprop_gradient = self.loss_prime(y_train_batch, train_output)
                for layer in reversed_layers:
                    backprop_gradient = layer.backward_propagation(backprop_gradient, learning_rate)
                
                if ((train_batch_index + 1) in [1, nb_train_batches]) or ((train_batch_index + 1) % train_batch_index_step == 0):
                    formatted_batch_index = format(train_batch_index + 1, train_batch_index_format)
                    
                    # clearing the currently printed row
                    print(blank_row_with_carriage_return, end="")
                    
                    train_batch_progress_row = f"{initial_spacing}epoch {formatted_epoch_index}/{nb_epochs}  -  batch {formatted_batch_index}/{nb_train_batches}\r"
                    print(train_batch_progress_row, end="")
            
            loss /= cast(nb_train_samples, utils.DEFAULT_DATATYPE)
            check_dtype(loss, utils.DEFAULT_DATATYPE)
            accuracy /= nb_train_samples
            
            # -------------------------------------------------------------- #
            
            # validation step for the current epoch
            
            val_loss     = cast(0.0, utils.DEFAULT_DATATYPE)
            val_accuracy = 0.0
            
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
            val_accuracy /= nb_val_samples
            
            # -------------------------------------------------------------- #
            
            # updating the network's history
            
            self.history["epoch"].append(epoch_index + 1)
            self.history["loss"].append(loss)
            self.history["val_loss"].append(val_loss)
            self.history["accuracy"].append(accuracy)
            self.history["val_accuracy"].append(val_accuracy)
            
            # -------------------------------------------------------------- #
            
            # clearing the currently printed row
            print(blank_row_with_carriage_return, end="")
            
            precision_epoch_history = 4
            epoch_history = f"{initial_spacing}epoch {formatted_epoch_index}/{nb_epochs}  -  loss={loss:.{precision_epoch_history}f}  -  val_loss={val_loss:.{precision_epoch_history}f}  -  accuracy={accuracy:.{precision_epoch_history}f}  -  val_accuracy={val_accuracy:.{precision_epoch_history}f}"
            print(epoch_history)
        
        # ================================================================== #
        
        t_end_training = time()
        duration_training = t_end_training - t_beginning_training
        
        average_epoch_duration = duration_training / nb_epochs
        average_batch_duration = average_epoch_duration / (nb_train_batches + nb_val_batches)
        average_batch_duration_in_milliseconds = 1000 * average_batch_duration
        
        conclusion = f"\n{initial_spacing}Training complete ! Done in {duration_training:.1f} seconds ({average_epoch_duration:.1f} s/epoch, {average_batch_duration_in_milliseconds:.1f} ms/batch)"
        print(conclusion)
        
        print(transition)
    
    
    def plot_history(
            self,
            save_plot_to_disk=False,
            saved_image_name="network_history"
        ):
        """
        Plots the evolution of the losses and the accuracies of the network
        during the (last) training phase
        """
        
        if self.history is None:
            print("\nNetwork.plot_history - WARNING : You cannot plot the network's history if you haven't trained it yet !")
            return
        
        epochs         = self.history["epoch"]
        losses         = self.history["loss"]
        val_losses     = self.history["val_loss"]
        accuracies     = self.history["accuracy"]
        val_accuracies = self.history["val_accuracy"]
        
        nb_epochs = len(epochs)
        if nb_epochs == 1:
            print("\nNetwork.plot_history - WARNING : You cannot plot the network's history if you only trained it on 1 epoch !")
            return
        
        # for example
        color_of_train_data = "dodgerblue"
        color_of_val_data   = "orange"
        
        # ------------------------------------------------------------------ #
        
        # generating the plot
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        plt.suptitle(f"\nHistory of the network's training phase (on {nb_epochs} epochs)", fontsize=15)
        
        # losses
        ax[0].plot(epochs, losses, color=color_of_train_data, label="loss")
        ax[0].plot(epochs, val_losses, color=color_of_val_data, label="val_loss")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True)) # force integer ticks on the x-axis (i.e. the epochs axis)
        max_loss_value = max(np.max(losses), np.max(val_losses))
        ax[0].set_ylim([0, 1.05 * max_loss_value])
        ax[0].legend()
        ax[0].set_title("Losses for the \"train\" and \"val\" datasets")
        
        # accuracies
        ax[1].plot(epochs, accuracies, color=color_of_train_data, label="accuracy")
        ax[1].plot(epochs, val_accuracies, color=color_of_val_data, label="val_accuracy")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True)) # force integer ticks on the x-axis (i.e. the epochs axis)
        ax[1].set_ylim([0, 1])
        ax[1].legend()
        ax[1].set_title("Accuracies for the \"train\" and \"val\" datasets")
        
        plt.subplots_adjust(top=0.85)
        
        # ------------------------------------------------------------------ #
        
        # saving the plot (if requested)
        
        if save_plot_to_disk:
            t_beginning_image_saving = time()
            
            # creating the folder containing the saved plot (if it doesn't exist)
            DEFAULT_SAVED_IMAGES_FOLDER_NAME = "saved_plots"
            if not(os.path.exists(DEFAULT_SAVED_IMAGES_FOLDER_NAME)):
                os.mkdir(DEFAULT_SAVED_IMAGES_FOLDER_NAME)
            
            if saved_image_name[-4 : ] != ".png": 
                saved_image_full_name = saved_image_name + ".png"
            else:
                saved_image_full_name = saved_image_name
            
            saved_image_path = os.path.join(
                DEFAULT_SAVED_IMAGES_FOLDER_NAME,
                saved_image_full_name
            )
            
            # actually saving the plot
            plt.savefig(saved_image_path, dpi=300, format="png")
            
            t_end_image_saving = time()
            duration_image_saving = t_end_image_saving - t_beginning_image_saving
            print(f"\nThe plot \"{saved_image_path}\" was successfully saved. Done in {duration_image_saving:.3f} seconds")
        
        # ------------------------------------------------------------------ #
        
        # actually displaying the plot
        plt.show()
    
    
    def predict(self, X_test, test_batch_size=32, return_logits=True):
        """
        Returns the network's raw prediction (i.e. the logits) for a given
        test input
        """
        
        if self.history is None:
            raise Exception("Network.predict - You haven't trained the network yet !")
        
        nb_test_samples = X_test.shape[0]
        
        # only used to split the testing data into batches
        dummy_y_test = np.zeros((nb_test_samples, ), dtype=X_test.dtype)
        
        test_batches = split_data_into_batches(
            X_test,
            dummy_y_test,
            batch_size=min(test_batch_size, nb_test_samples),
            normalize_batches=self.__normalize_input_data,
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
        y_pred = np.array(np.vstack(tuple(test_outputs)), dtype=X_test.dtype)
        
        if not(return_logits):
            # in this case, the 1D vector of INTEGER labels is returned
            y_pred = categorical_to_vector(y_pred)
        
        return y_pred
    
    
    def evaluate(self, X_test, y_test, test_batch_size=32):
        """
        Computes the network's prediction of `X_test` (i.e. `y_pred`), then
        computes the accuracy score and the confusion matrix of (y_test, y_pred)
        
        Here, `y_test` can either be a 1D vector of INTEGER labels or its
        one-hot encoded equivalent (in that case it will be a 2D matrix)
        """
        
        if self.history is None:
            raise Exception("Network.evaluate - You haven't trained the network yet !")
        
        if len(y_test.shape) == 1:
           y_test_flat = y_test.copy()
        else:
            # here, `y_test` is a one-hot encoded matrix
            assert len(y_test.shape) == 2
            y_test_flat = categorical_to_vector(y_test)
        
        # here we don't need the logits, we only need the predicted (1D) vector
        # of INTEGER labels
        y_pred_flat = self.predict(
            X_test,
            test_batch_size=test_batch_size,
            return_logits=False
        )
        assert len(y_pred_flat.shape) == 1
        
        acc_score = 100 * accuracy_score(y_test_flat, y_pred_flat)
        conf_matrix = confusion_matrix(y_test_flat, y_pred_flat)
        
        # checking that the accuracy score computed from the confusion matrix
        # is the same as the one computed by the `accuracy_score` function
        acc_score_from_conf_matrix = 100 * float(np.sum(np.diag(conf_matrix))) / y_test_flat.size
        assert np.allclose(acc_score, acc_score_from_conf_matrix)
        
        return acc_score, conf_matrix
    
    
    def display_some_predictions(self, X_test, y_test, seed=None):
        """
        Displays predictions of random test samples
        
        Here, `y_test` can either be a 1D vector of INTEGER labels or its
        one-hot encoded equivalent (in that case it will be a 2D matrix)
        """
        
        if self.history is None:
            raise Exception("Network.display_some_predictions - You haven't trained the network yet !")
        
        nb_rows    = 2
        nb_columns = 5
        
        nb_predictions = nb_rows * nb_columns
        nb_test_samples = X_test.shape[0]
        nb_pixels_per_image = X_test.shape[1]
        
        # here we're assuming that the images are square-shaped
        sidelength_of_each_image = int(round(np.sqrt(nb_pixels_per_image)))
        default_image_size = (sidelength_of_each_image, sidelength_of_each_image)
        
        if len(y_test.shape) == 1:
            y_test_flat = y_test.copy()
        else:
            # here, `y_test` is a one-hot encoded matrix
            assert len(y_test.shape) == 2
            y_test_flat = categorical_to_vector(y_test)
        
        # ------------------------------------------------------------------ #
        
        # doing the predictions of the random test samples
        
        np.random.seed(seed)
        random_test_indices = np.random.choice(np.arange(nb_test_samples), size=(nb_predictions, ))
        np.random.seed(None) # resetting the seed
        
        random_test_samples = X_test[random_test_indices, :]
        logits = self.predict(random_test_samples)
        predicted_digit_values = categorical_to_vector(logits)
        
        # by default
        confidence_level_precision = 0
        
        # in order to actually get probability distributions, we're normalizing
        # the logits (in case we're using a final activation function that isn't
        # softmax, like sigmoid for instance)
        logits /= np.sum(logits, axis=1, keepdims=True)
        
        # ------------------------------------------------------------------ #
        
        # displaying the predictions
        
        fig, ax = plt.subplots(nb_rows, nb_columns, figsize=(16, 8))
        plt.suptitle(f"\nPredictions of {nb_predictions}/{nb_test_samples} random test samples (and their confidence level)", fontsize=15)
        
        for image_index in range(nb_predictions):
            predicted_digit_value = predicted_digit_values[image_index]
            confidence_level_first_choice = 100 * logits[image_index, predicted_digit_value]
            
            second_choice_predicted_digit = np.argsort(logits[image_index, :])[::-1][1]
            confidence_level_second_choice = 100 * logits[image_index, second_choice_predicted_digit]
            
            actual_digit_value = y_test_flat[random_test_indices[image_index]]
            
            random_test_image_2D = random_test_samples[image_index, :].reshape(default_image_size)
            
            row_index = image_index // nb_columns
            column_index = image_index % nb_columns
            ax[row_index, column_index].imshow(random_test_image_2D, cmap="gray")
            
            subplot_title = f"1st prediction : {predicted_digit_value} ({confidence_level_first_choice:.{confidence_level_precision}f}%)"
            subplot_title += f"\n2nd prediction : {second_choice_predicted_digit} ({confidence_level_second_choice:.{confidence_level_precision}f}%)"
            subplot_title += f"\nActual value : {actual_digit_value}"
            
            ax[row_index, column_index].set_title(subplot_title)
            ax[row_index, column_index].axis("off")
        
        plt.subplots_adjust(wspace=0.5, top=0.85)
        
        plt.show()

