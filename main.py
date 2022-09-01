# -*- coding: utf-8 -*-

"""
Script implementing a Multi-Layer Perceptron (MLP) from scratch
"""

from mnist_dataset import (
    load_raw_MNIST_dataset,
    plot_random_images_from_raw_MNIST_dataset,
    format_raw_MNIST_dataset
)

from layers import (
    InputLayer,
    DenseLayer,
    ActivationLayer,
    BatchNormLayer,
    DropoutLayer
)

from network import Network

from utils import print_confusion_matrix


##############################################################################


# DEBUGGING

if __name__ == "__main__":
    # ====================================================================== #
    
    # Loading and formatting the data
    
    # This seed is currently used to :
    #     1) Randomly split the raw data into the "train", "val" and "test" sets
    #     2) Randomly shuffle the "train", "val" and "test" sets
    # Set this seed to `None` for "real" randomness during those 2 processes
    seed_data_formatting = 555
    
    # Defining the number of samples in the "train", "val" and "test" sets
    # NB : The validation set is extracted from the raw "train" data, not from
    #      the raw "test" data. As a reminder, there are :
    #          - 60000 samples in the raw "train" data
    #          - 10000 samples in the raw "test" data
    nb_train_samples = 10000
    nb_val_samples   = 1000
    nb_test_samples  = 1000
    
    raw_X_train, raw_y_train, raw_X_test, raw_y_test = load_raw_MNIST_dataset(
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
    
    X_train, y_train, X_val, y_val, X_test, y_test = format_raw_MNIST_dataset(
        raw_X_train,
        raw_y_train,
        raw_X_test,
        raw_y_test,
        nb_train_samples,
        nb_val_samples,
        nb_test_samples,
        nb_shuffles=20,
        seed=seed_data_formatting,
        verbose=False
    )
    
    # ====================================================================== #
    
    # Defining the input and output sizes of the network (respectively)
    
    # = 28 * 28 = 784 pixels per image
    nb_pixels_per_image = X_train.shape[1]
    
    # = 10 digits
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
    
    nb_epochs = 10
    learning_rate = 0.15
    
    train_batch_size = 40
    
    nb_neurons_hidden_dense_layers = [
        256,
        64,
        32
    ]
    
    use_batch_norm_layers = False
    
    use_dropout_layers = False
    if use_dropout_layers:
        # The Dropout layer randomly sets input units to 0 with a frequency of
        # `dropout_rate` at each step during the training phase. This regularization
        # method helps prevent overfitting
        dropout_rate = 0.10
    
    # ====================================================================== #
    
    # Building the MLP network architecture from the previously defined
    # hyperparameters
    
    # ---------------------------------------------------------------------- #
    
    # Initializing the network
    
    # If you set `normalize_input_data` to `True`, every time the data will
    # be split into batches (during the training, validation AND testing phases),
    # each resulting batch will be normalized such that its mean is 0 and its
    # standard deviation is 1. It's HIGHLY recommended to set `normalize_input_data`
    # to `True` here, in order to have better performances
    network = Network(normalize_input_data=True)
    
    # Input layer
    network.add(InputLayer(input_size=nb_pixels_per_image))
    
    # ---------------------------------------------------------------------- #
    
    # Hidden layers
    
    for hidden_layer_index, nb_neurons in enumerate(nb_neurons_hidden_dense_layers):
        if seed_network is not None:
            # updating the seed such that the "randomness" in the added
            # Dense/Dropout layers is different each time
            seed = seed_network + hidden_layer_index
        else:
            seed = None
        
        network.add(DenseLayer(nb_neurons, seed=seed))
        
        if use_batch_norm_layers:
            # Adding a BatchNorm regularization layer (if requested)
            network.add(BatchNormLayer())
        
        """
        Possible relevant choices here (the activation name is case insensitive) :
            network.add(ActivationLayer("ReLU"))
            OR
            network.add(ActivationLayer("leaky_ReLU", leaky_ReLU_coeff=0.01))
            OR
            network.add(ActivationLayer("tanh"))
        """
        network.add(ActivationLayer("ReLU"))
        
        if use_dropout_layers:
            # Adding a dropout regularization layer (if requested)
            network.add(DropoutLayer(dropout_rate, seed=seed))
    
    # ---------------------------------------------------------------------- #
    
    # Output layers
    
    if seed_network is not None:
        # updating the seed (for the same reasons as before)
        seed = seed_network + len(nb_neurons_hidden_dense_layers)
    else:
        seed = None
    
    network.add(DenseLayer(nb_classes, seed=seed))
    
    """
    Possible relevant choices here (the activation name is case insensitive) :
        network.add(ActivationLayer("softmax"))
        OR
        network.add(ActivationLayer("sigmoid"))
    """
    network.add(ActivationLayer("softmax"))
    
    # ---------------------------------------------------------------------- #
    
    # Displaying the summary of the network's architecture
    
    # NB : The kwargs of this method will only affect how the summary will look
    #      like when it's printed (they won't affect the summary's contents)
    network.summary(
        initial_spacing=1,
        column_separator="|", # can be multiple characters long
        row_separator="-",    # has to be a single character
        bounding_box="*"      # has to be a single character
    )
    
    # Or, equivalently, you can run : `print(network)`
    
    # ====================================================================== #
    
    # Setting the loss function of the network
    
    """
    Possible relevant choices here (the loss function name is case insensitive) :
        network.set_loss_function("CCE") # CCE = Categorical Cross-Entropy
        OR
        network.set_loss_function("MSE") # MSE = Mean Squared Error
    """
    network.set_loss_function("CCE")
    
    # ====================================================================== #
    
    # Training phase
    
    training_data = (X_train, y_train)
    validation_data = (X_val, y_val)
    
    network.fit(
        training_data,
        validation_data,
        nb_epochs,
        learning_rate,
        train_batch_size,
        nb_shuffles_before_train_batch_splits=10,
        seed_train_batch_splits=seed_network,
        val_batch_size=32
    )
    
    # ====================================================================== #
    
    # RESULTS
    
    network.plot_history(
        save_plot_to_disk=False,
        saved_image_name="network_history" # it will be saved as a PNG image by default
    )
    
    acc_score, conf_matrix = network.evaluate(
        X_test,
        y_test,
        test_batch_size=32
    )
    
    print_confusion_matrix(
        conf_matrix,
        normalize="rows", # = "rows", "columns" or "no"
        precision=1,
        initial_spacing=1,
        display_with_line_breaks=True
    )
    
    precision_global_accuracy = 2 # by default
    print(f"\nGLOBAL ACCURACY : {acc_score:.{precision_global_accuracy}f} %\n")
    
    # Just for testing purposes
    network.display_some_predictions(
        X_test,
        y_test,
        seed=None
    )
    
    # ===============================  END  =============================== #

