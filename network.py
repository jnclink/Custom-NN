# -*- coding: utf-8 -*-

"""
Main neural network class
"""

import sys, os
from time import time
import numpy as np
import matplotlib.pyplot as plt

# used to force integer ticks on the x-axis of a plot
from matplotlib.ticker import MaxNLocator

from utils import (
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
    DropoutLayer
)


##############################################################################


class Network:
    """
    Main network/model class
    """
    
    # class variable
    AVAILABLE_LOSSES = {
        "mse" : (MSE, MSE_prime), # MSE = Mean Square Error
        "cce" : (CCE, CCE_prime)  # CCE = Categorical Cross-Entropy
    }
    
    def __init__(self, normalize_input_data=True):
        self.layers = []
        self.loss_name = None
        self.loss = None
        self.loss_prime = None
        self.history = None
        
        # list of all the input/output sizes of the network
        self._sizes = []
        
        self.__normalize_input_data = normalize_input_data
    
    
    def __str__(self):
        # using the default summary kwargs (except for `print_summary`, which
        # has to be set to `False` here, in order to prevent a duplicate print
        # of the summary)
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
        if isinstance(layer, InputLayer):
            assert len(self._sizes) == 0
            input_size  = layer.input_size
            output_size = input_size
        else:
            assert len(self._sizes) >= 1, "\nPlease add an Input layer to the network first !"
            input_size  = self._sizes[-1][1] # output size of the previous layer
            
            if isinstance(layer, DenseLayer):
                output_size = layer.output_size
                layer.build(input_size)
            elif isinstance(layer, (ActivationLayer, DropoutLayer)):
                output_size = input_size
            else:
                print(f"\nNetwork.add - ERROR - Unrecognized layer type : \"{layer.__class__.__name__}\"")
                sys.exit(-1)
        
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
        Gets the raw data that will be printed in the summary. The reason we
        collect the entire summary data before printing it is to align the
        columns
        """
        summary_data = {
            "layer_types"      : ["Layer"],
            "input_shapes"     : ["Input shape"],
            "output_shapes"    : ["Output shape"],
            "trainable_params" : ["Trainable parameters"]
        }
        
        for layer_index, layer in enumerate(self.layers):
            layer_type = str(layer).replace("Layer", "")
            
            input_shape  = str((None, self._sizes[layer_index][0]))
            output_shape = str((None, self._sizes[layer_index][1]))
            
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
        
        assert len(self.layers) >= 1
        
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
        
        return str_summary
    
    
    def set_loss_function(self, loss_name):
        """
        Sets the loss function of the network
        """
        assert isinstance(loss_name, str)
        loss_name = loss_name.lower()
        if loss_name not in list(self.AVAILABLE_LOSSES.keys()):
            print(f"\nNetwork.set_loss_function - ERROR - Unrecognized loss function name : \"{loss_name}\"")
            sys.exit(-1)
        
        self.loss_name = loss_name
        self.loss, self.loss_prime = self.AVAILABLE_LOSSES[self.loss_name]
    
    
    def fit(
            self,
            training_data,
            validation_data,
            nb_epochs,
            learning_rate,
            train_batch_size,
            nb_shuffles_before_train_batch_split=10,
            seed_train_batch_split=None,
            val_batch_size=32
        ):
        """
        Trains the network on `nb_epochs` epochs
        """
        
        if (self.loss is None) or (self.loss_prime is None):
            print("\nNetwork.fit - ERROR : Please set a loss function before training the network !")
            sys.exit(-1)
        
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
        
        # ================================================================== #
        
        # training loop
        
        t_beginning_training = time()
        
        print(transition)
        print(f"\n{initial_spacing}Starting the training loop ...\n")
        
        for epoch_index in range(1, nb_epochs + 1):
            train_batches = split_data_into_batches(
                X_train,
                y_train,
                train_batch_size,
                normalize_batches=self.__normalize_input_data,
                nb_shuffles=nb_shuffles_before_train_batch_split,
                seed=seed_train_batch_split
            )
            
            # for display purposes only
            formatted_epoch_index = format(epoch_index, epoch_index_format)
            
            loss = 0.0
            accuracy = 0.0
            
            for train_batch_index in range(1, nb_train_batches + 1):
                X_train_batch = train_batches["data"][train_batch_index - 1]
                y_train_batch = train_batches["labels"][train_batch_index - 1]
                
                # forward propagation
                train_output = X_train_batch
                for layer in self.layers:
                    train_output = layer.forward_propagation(train_output)
                
                # compute the loss and accuracy (for display purposes only)
                loss += np.sum(self.loss(y_train_batch, train_output))
                accuracy += accuracy_score(
                    categorical_to_vector(y_train_batch),
                    categorical_to_vector(train_output),
                    normalize=False
                )
                
                # backward propagation
                backprop_error = self.loss_prime(y_train_batch, train_output)
                for layer in reversed(self.layers):
                    backprop_error = layer.backward_propagation(backprop_error, learning_rate)
                
                if (train_batch_index in [1, nb_train_batches]) or (train_batch_index % train_batch_index_step == 0):
                    formatted_batch_index = format(train_batch_index, train_batch_index_format)
                    
                    # clear the currently printed row
                    print(" " * 150 + "\r", end="")
                    
                    train_batch_progress_row = f"{initial_spacing}epoch {formatted_epoch_index}/{nb_epochs}  -  batch {formatted_batch_index}/{nb_train_batches}\r"
                    print(train_batch_progress_row, end="")
            
            loss /= nb_train_samples
            accuracy /= nb_train_samples
            
            # -------------------------------------------------------------- #
            
            # validation step for the current epoch
            
            val_loss = 0.0
            val_accuracy = 0.0
            
            for val_batch_index in range(nb_val_batches):
                X_val_batch = val_batches["data"][val_batch_index]
                y_val_batch = val_batches["labels"][val_batch_index]
                
                # forward propagation
                val_output = X_val_batch
                for layer in self.layers:
                    val_output = layer.forward_propagation(val_output, training=False)
                
                # compute the validation loss and accuracy (for display purposes only)
                val_loss += np.sum(self.loss(y_val_batch, val_output))
                val_accuracy += accuracy_score(
                    categorical_to_vector(y_val_batch),
                    categorical_to_vector(val_output),
                    normalize=False
                )
            
            val_loss /= nb_val_samples
            val_accuracy /= nb_val_samples
            
            # -------------------------------------------------------------- #
            
            # updating the network's history
            
            self.history["epoch"].append(epoch_index)
            self.history["loss"].append(loss)
            self.history["val_loss"].append(val_loss)
            self.history["accuracy"].append(accuracy)
            self.history["val_accuracy"].append(val_accuracy)
            
            # -------------------------------------------------------------- #
            
            # clear the currently printed row
            print(" " * 150 + "\r", end="")
            
            precision = 4
            epoch_history = f"{initial_spacing}epoch {formatted_epoch_index}/{nb_epochs}  -  loss={loss:.{precision}f}  -  val_loss={val_loss:.{precision}f}  -  accuracy={accuracy:.{precision}f}  -  val_accuracy={val_accuracy:.{precision}f}"
            print(epoch_history)
        
        # ================================================================== #
        
        t_end_training = time()
        duration_training = t_end_training - t_beginning_training
        
        average_epoch_duration = duration_training / nb_epochs
        average_batch_duration = average_epoch_duration / nb_train_batches
        if average_batch_duration < 1:
            precision_average_batch_duration = 3
        else:
            precision_average_batch_duration = 1
        
        print(f"\n{initial_spacing}Training complete ! Done in {duration_training:.1f} seconds ({average_epoch_duration:.1f} s/epoch, {average_batch_duration:.{precision_average_batch_duration}f} s/batch)")
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
            if saved_image_name[-4 : ] != ".png": 
                saved_image_full_name = saved_image_name + ".png"
            else:
                saved_image_full_name = saved_image_name
            
            DEFAULT_SAVED_IMAGES_FOLDER = "saved_plots"
            if not(os.path.exists(DEFAULT_SAVED_IMAGES_FOLDER)):
                os.mkdir(DEFAULT_SAVED_IMAGES_FOLDER)
            
            saved_image_path = os.path.join(
                DEFAULT_SAVED_IMAGES_FOLDER,
                saved_image_full_name
            )
            
            t_beginning_image_saving = time()
            
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
            print("\nNetwork.predict - ERROR : You haven't trained the network yet !")
            sys.exit(-1)
        
        nb_test_samples = X_test.shape[0]
        
        # only used to split the testing data into batches
        dummy_y_test = np.zeros((nb_test_samples, 1), dtype=X_test.dtype)
        
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
        """
        
        if self.history is None:
            print("\nNetwork.evaluate - ERROR : You haven't trained the network yet !")
            sys.exit(-1)
        
        if len(y_test.shape) == 1:
           y_test_flat = y_test.copy()
        else:
            # here, `y_true` is a one-hot encoded matrix
            assert len(y_test.shape) == 2
            y_test_flat = categorical_to_vector(y_test)
        
        y_pred_flat = self.predict(
            X_test,
            test_batch_size=test_batch_size,
            return_logits=False
        )
        assert len(y_pred_flat.shape) == 1
        
        acc_score = 100 * accuracy_score(y_test_flat, y_pred_flat)
        conf_matrix = confusion_matrix(y_test_flat, y_pred_flat)
        
        return acc_score, conf_matrix
    
    
    def display_some_predictions(self, X_test, y_test, seed=None):
        if self.history is None:
            print("\nNetwork.display_some_predictions - ERROR : You haven't trained the network yet !")
            sys.exit(-1)
        
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
        
        fig, ax = plt.subplots(nb_rows, nb_columns, figsize=(12, 8))
        plt.suptitle(f"\nPredictions of {nb_predictions}/{nb_test_samples} random test samples", fontsize=15)
        
        np.random.seed(seed)
        
        for image_index in range(nb_predictions):
            random_test_index = np.random.randint(0, nb_test_samples)
            
            # prediction of 1 random test image/sample
            random_test_image = X_test[random_test_index, :].reshape((1, nb_pixels_per_image))
            predicted_digit_value = self.predict(random_test_image, return_logits=False)[0]
            
            actual_digit_value = y_test_flat[random_test_index]
            
            random_test_image_2D = random_test_image.reshape(default_image_size)
            
            row_index = image_index // nb_columns
            column_index = image_index % nb_columns
            ax[row_index, column_index].imshow(random_test_image_2D, cmap="gray")
            ax[row_index, column_index].set_title(f"Predicted : {predicted_digit_value}\nActual : {actual_digit_value}")
        
        # resetting the seed
        np.random.seed(None)
        
        plt.show()

