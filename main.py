# -*- coding: utf-8 -*-

"""
Script debugging the whole custom Multi-Layer Perceptron (MLP) implementation
"""

from utils import (
    set_global_datatype,
    print_confusion_matrix
)

from mnist_dataset import (
    load_raw_MNIST_dataset_from_disk,
    plot_random_images_from_raw_MNIST_dataset,
    format_raw_MNIST_dataset
)

from network import Network

from layers import (
    InputLayer,
    DenseLayer,
    ActivationLayer,
    BatchNormLayer,
    DropoutLayer
)


##############################################################################


def main():
    """
    Main debugging function
    """
    
    # ====================================================================== #
    
    # Defining the datatype of ALL the data that will flow through the network
    
    # = "float32" (default) or "float64"
    datatype = "float32"
    
    set_global_datatype(datatype)
    
    # ====================================================================== #
    
    # Loading and formatting the data
    
    # This seed is currently used to :
    #     1) Randomly split the raw data into the "train", "val" and "test" sets
    #     2) Randomly shuffle the "train", "val" and "test" sets
    # Set this seed to `None` for "real" randomness during those 2 processes
    seed_data_formatting = 555
    
    # Here, `selected_classes` can either be :
    #     - The string "all", if you want to work with all the digits ranging
    #       from 0 to 9 (default)
    #     - A list/tuple/1D-array containing the specific digits you want to
    #       work with (e.g. [2, 4, 7])
    selected_classes = "all"
    
    # Defining the number of samples in the "train", "val" and "test" sets
    # NB : The validation set is extracted from the raw "train" data, not from
    # the raw "test" data. As a reminder, there are :
    #     - 60000 samples in the raw "train" data (if ALL the classes are selected)
    #     - 10000 samples in the raw "test" data (if ALL the classes are selected)
    nb_train_samples = 10000
    nb_val_samples   = 1000 # can be set to zero if needed
    nb_test_samples  = 1000
    
    raw_X_train, raw_y_train, raw_X_test, raw_y_test = load_raw_MNIST_dataset_from_disk(
        verbose=False
    )
    
    plot_random_images = False
    if plot_random_images:
        plot_random_images_from_raw_MNIST_dataset(
            raw_X_train,
            raw_y_train,
            raw_X_test,
            raw_y_test,
            seed=None
        )
    
    # NB : If you set `nb_val_samples` to zero, `X_val` and `y_val` will both
    #      be equal to `None`
    X_train, y_train, X_val, y_val, X_test, y_test = format_raw_MNIST_dataset(
        raw_X_train,
        raw_y_train,
        raw_X_test,
        raw_y_test,
        nb_train_samples,
        nb_val_samples,
        nb_test_samples,
        selected_classes=selected_classes,
        nb_shuffles=20,
        seed=seed_data_formatting,
        verbose=False
    )
    
    # ====================================================================== #
    
    # Defining the input and output sizes of the network (respectively)
    
    # = 28 * 28 = 784 pixels per image
    nb_pixels_per_image = X_train.shape[1]
    
    # = number of (distinct) selected digits
    nb_classes = y_train.shape[1]
    
    # ====================================================================== #
    
    # Defining the hyperparameters of the Multi-Layer Perceptron (MLP) network
    
    # This seed is currently used to :
    #     1) Randomly initialize the weights and biases of the Dense layers
    #     2) Randomly generate the dropout matrices of the Dropout layers (if
    #        these layers are used)
    #     3) Randomly split the training data into batches during the training
    #        phase (at each epoch)
    # Set this seed to `None` for "real" randomness during those 3 processes
    seed_network = 7777
    
    # Number of times the trainable parameters will be updated using the WHOLE
    # training data
    nb_epochs = 10
    
    # If you lower the batch size, you might also want to lower the learning
    # rate (to prevent the network from overfitting)
    train_batch_size = 40
    
    # The learning rate has to lie in the range ]0, 1[
    learning_rate = 0.15
    
    # In chronological order
    nb_neurons_in_hidden_dense_layers = [
        256,
        64,
        32
    ]
    
    # Relevant choices here : "ReLU", "leaky_ReLU" or "tanh". The main
    # activation name is case insensitive
    main_activation_name = "ReLU"
    
    if main_activation_name.lower() == "leaky_relu":
        # Defining the "leaky ReLU coefficient" (default value : 0.01). It has
        # to lie in the range ]0, 1[
        activation_kwargs = {
            "leaky_ReLU_coeff" : 0.01
        }
    else:
        activation_kwargs = {}
    
    # Relevant choices here : "softmax" or "sigmoid". The output activation
    # name is case insensitive
    output_activation_name = "softmax"
    
    # The BatchNorm layer is a regularization layer that helps prevent overfitting
    # (without necessarily improving the overall accuracy of the network). It
    # basically standardizes (i.e. it normalizes with a mean of 0 and a standard
    # deviation of 1) the outputs of the previous layer, and then applies an affine
    # transform to the standardized outputs. The 2 parameters of the affine
    # transform (typically called "gamma" and "beta") are the trainable parameters
    # of the layer
    use_batch_norm_layers = False
    
    # Just like the BatchNorm layer, the Dropout layer is a regularization layer
    # that helps prevent overfitting, without necessarily improving the overall
    # accuracy of the network. Basically, it randomly sets input values to 0 with
    # a frequency of `dropout_rate` at each step during the training phase. This
    # layer doesn't have any trainable parameters
    use_dropout_layers = False
    dropout_rate = 0.30 # has to lie in the range ]0, 1[
    
    # ====================================================================== #
    
    # Building the MLP network architecture from the previously defined
    # hyperparameters
    
    # ---------------------------------------------------------------------- #
    
    # Initializing the network
    
    # If you set the `standardize_input_data` kwarg to `True`, the training,
    # validation AND testing sets will be normalized such that their mean is 0
    # and their standard deviation is 1 (i.e. they will be standardized). It's
    # HIGHLY recommended to set `normalize_input_data` to `True` here, in order
    # to obtain better results
    network = Network(standardize_input_data=True)
    
    # Input layer
    network.add(InputLayer(input_size=nb_pixels_per_image))
    
    # ---------------------------------------------------------------------- #
    
    # Hidden layers
    
    seed = seed_network
    
    for hidden_layer_index, nb_neurons in enumerate(nb_neurons_in_hidden_dense_layers):
        network.add(DenseLayer(nb_neurons, seed=seed))
        
        if use_batch_norm_layers:
            # Adding a BatchNorm regularization layer (if requested)
            network.add(BatchNormLayer())
        
        network.add(ActivationLayer(main_activation_name, **activation_kwargs))
        
        if use_dropout_layers:
            # Adding a Dropout regularization layer (if requested)
            network.add(DropoutLayer(dropout_rate, seed=seed))
        
        if seed is not None:
            # Updating the seed such that the "randomness" in the added
            # Dense/Dropout layers is different each time (in case 2
            # consecutive values of `nb_neurons` are the same)
            seed += 1
    
    # ---------------------------------------------------------------------- #
    
    # Output layers
    
    if seed_network is not None:
        assert seed == seed_network + len(nb_neurons_in_hidden_dense_layers)
    
    network.add(DenseLayer(nb_classes, seed=seed))
    network.add(ActivationLayer(output_activation_name))
    
    # ---------------------------------------------------------------------- #
    
    # Displaying the summary of the network's architecture
    
    # NB : The kwargs of this method will only affect how the summary will look
    #      like when it's printed (they won't affect the summary's contents)
    network.summary(
        column_separator="|",        # can be multiple characters long
        row_separator="-",           # has to be a single character
        bounding_box="*",            # has to be a single character
        alignment="left",            # = "left" (default), "right" or "center"
        transition_row_style="full", # = "full" (default) or "partial"
        column_spacing=2,
        horizontal_spacing=4,
        vertical_spacing=1,
        offset_spacing=1,
    )
    
    # Or, equivalently, you can run `network.summary()` or `print(network)`
    
    # ====================================================================== #
    
    # Setting the loss function of the network
    
    # Relevant choices here : "CCE" (Categorical Cross-Entropy) or "MSE" (Mean
    # Squared Error). The loss function name is case insensitive
    loss_function_name = "CCE"
    
    network.set_loss_function(loss_function_name)
    
    # ====================================================================== #
    
    # Training phase
    
    # NB : Here, inputting validation data is optional. If you don't want to
    #      use validation data, please (at least) set the `validation_data`
    #      kwarg to `None` (or don't specify it at all). If you set `nb_val_samples`
    #      to zero, the validation kwargs will automatically be discarded
    
    if nb_val_samples > 0:
        validation_kwargs = {
            "validation_data" : (X_val, y_val),
            "val_batch_size"  : 32
        }
    else:
        validation_kwargs = {}
    
    network.fit(
        X_train,
        y_train,
        nb_epochs,
        train_batch_size,
        learning_rate,
        nb_shuffles_before_each_train_batch_split=10,
        seed_train_batch_splits=seed_network,
        **validation_kwargs
    )
    
    # ====================================================================== #
    
    # RESULTS
    
    # ---------------------------------------------------------------------- #
    
    # Plotting the network's history
    
    network.plot_history(
        save_plot_to_disk=False,
        saved_image_name="network_history" # it will be saved as a PNG image by default
    )
    
    # ---------------------------------------------------------------------- #
    
    # Computing the global accuracy scores, the testing loss and the (raw)
    # confusion matrix of the network
    
    # The "top-N accuracy" is defined as the proportion of the true classes
    # that lie within the `N` most probable predicted classes (here, `N` is
    # actually `top_N_accuracy`)
    top_N_accuracy = 2
    
    acc_score, top_N_acc_score, test_loss, conf_matrix = network.evaluate(
        X_test,
        y_test,
        top_N_accuracy=top_N_accuracy,
        test_batch_size=32
    )
    
    # ---------------------------------------------------------------------- #
    
    # Displaying the confusion matrices of the network
    
    for normalize in ["no", "columns", "rows"]:
        print_confusion_matrix(
            conf_matrix,
            selected_classes=selected_classes,
            normalize=normalize, # = "columns" (default), "rows" or "no"
            precision=1,
            color="green", # = "green" (default), "blue", "purple", "red" or "orange"
            offset_spacing=1,
            display_with_line_breaks=True
        )
    
    # ---------------------------------------------------------------------- #
    
    # Displaying the testing loss and the global accuracy scores of the network
    
    precision_loss = 4 # by default
    print(f"\nTESTING LOSS    : {test_loss:.{precision_loss}f}")
    
    precision_accuracy = 2 # by default
    print(f"\nGLOBAL ACCURACY : {acc_score:.{precision_accuracy}f} %")
    potential_extra_space = " " * int(top_N_accuracy < 10)
    print(f"TOP-{top_N_accuracy}{potential_extra_space} ACCURACY : {top_N_acc_score:.{precision_accuracy}f} %\n")
    
    # ---------------------------------------------------------------------- #
    
    # Displaying some of the network's predictions (for testing purposes only)
    
    network.display_some_predictions(
        X_test,
        y_test,
        selected_classes=selected_classes,
        seed=None
    )
    
    # ===============================  END  =============================== #


##############################################################################


# DEBUGGING

if __name__ == "__main__":
    main()

