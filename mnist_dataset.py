# -*- coding: utf-8 -*-

"""
Script defining the functions used to load and format the raw MNIST dataset
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

# only used to load the raw MNIST dataset
from tensorflow.keras.datasets import mnist

# only used to split the data, such that the class distribution of the split
# data is (roughly) the same as the class distribution of the initial raw data
# (this is done using the VERY USEFUL `stratify` kwarg of the `train_test_split`
# function)
from sklearn.model_selection import train_test_split

import utils
from utils import (
    cast,
    check_dtype,
    get_dtype_of_array,
    get_range_of_array,
    to_categorical,
    categorical_to_vector,
    check_if_label_vector_is_valid,
    list_to_string,
    display_class_distributions
)


##############################################################################


def load_raw_MNIST_dataset(verbose=False):
    """
    This function is basically a wrapper of the `mnist.load_data` function
    """
    assert isinstance(verbose, bool)
    
    t_beginning_loading = time()
    
    # actually loading the raw data from Keras
    (raw_X_train, raw_y_train), (raw_X_test, raw_y_test) = mnist.load_data()
    
    t_end_loading = time()
    duration_loading = t_end_loading - t_beginning_loading
    
    if verbose:
        print("\nShapes of the raw MNIST data (loaded from Keras) :")
        print(f"    - X_train : {raw_X_train.shape}")
        print(f"    - y_train : {raw_y_train.shape}")
        print(f"    - X_test  : {raw_X_test.shape}")
        print(f"    - y_test  : {raw_y_test.shape}")
        
        print("\nTypes of the raw MNIST data (loaded from Keras) :")
        print(f"    - X_train : {get_dtype_of_array(raw_X_train)}")
        print(f"    - y_train : {get_dtype_of_array(raw_y_train)}")
        print(f"    - X_test  : {get_dtype_of_array(raw_X_test)}")
        print(f"    - y_test  : {get_dtype_of_array(raw_y_test)}")
        
        precision_of_printed_info = 3
        print("\nRanges of the raw MNIST data (loaded from Keras) :")
        print(f"    - X_train : {get_range_of_array(raw_X_train, precision=precision_of_printed_info)}")
        print(f"    - y_train : {get_range_of_array(raw_y_train, precision=precision_of_printed_info)}")
        print(f"    - X_test  : {get_range_of_array(raw_X_test,  precision=precision_of_printed_info)}")
        print(f"    - y_test  : {get_range_of_array(raw_y_test,  precision=precision_of_printed_info)}")
        
        # displaying the class distribution of the raw MNIST data
        dict_of_label_vectors = {
            "y_train" : raw_y_train,
            "y_test"  : raw_y_test
        }
        display_class_distributions(
            dict_of_label_vectors,
            selected_classes="all",
            precision=2
        )
    
    print(f"\nThe raw MNIST dataset was successfully loaded. Done in {duration_loading:.3f} seconds")
    
    return raw_X_train, raw_y_train, raw_X_test, raw_y_test


##############################################################################


def _validate_raw_MNIST_data(
        raw_X_train,
        raw_y_train,
        raw_X_test,
        raw_y_test
    ):
    """
    Checks if the specified raw MNIST data is valid or not. By design, the
    arguments `raw_X_train`, `raw_y_train`, `raw_X_test` and `raw_y_test` are
    meant to be the outputs of the `load_raw_MNIST_dataset` function of this
    script
    """
    assert isinstance(raw_X_train, np.ndarray)
    assert raw_X_train.shape == (60000, 28, 28)
    check_dtype(raw_X_train, np.uint8)
    
    assert isinstance(raw_y_train, np.ndarray)
    assert raw_y_train.shape == (60000, )
    check_dtype(raw_y_train, np.uint8)
    
    assert isinstance(raw_X_test, np.ndarray)
    assert raw_X_test.shape == (10000, 28, 28)
    check_dtype(raw_X_test, np.uint8)
    
    assert isinstance(raw_y_test, np.ndarray)
    assert raw_y_test.shape == (10000, )
    check_dtype(raw_y_test, np.uint8)
    
    expected_classes = np.arange(10)
    assert np.allclose(np.unique(raw_y_train), expected_classes)
    assert np.allclose(np.unique(raw_y_test),  expected_classes)


##############################################################################


def plot_random_images_from_raw_MNIST_dataset(
        raw_X_train,
        raw_y_train,
        raw_X_test,
        raw_y_test,
        seed=None
    ):
    """
    Plots a random sample of each of the 10 digits (from the raw MNIST data).
    By design, the arguments `raw_X_train`, `raw_y_train`, `raw_X_test` and
    `raw_y_test` are meant to be the outputs of the `load_raw_MNIST_dataset`
    function of this script
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    _validate_raw_MNIST_data(
        raw_X_train,
        raw_y_train,
        raw_X_test,
        raw_y_test
    )
    
    assert isinstance(seed, (type(None), int))
    if isinstance(seed, int):
        assert seed >= 0
    
    # ---------------------------------------------------------------------- #
    
    nb_rows    = 2
    nb_columns = 5
    
    nb_classes = np.unique(raw_y_train).size # = 10
    assert nb_classes >= 2
    assert nb_rows * nb_columns == nb_classes
    
    data_types = ["train", "test"]
    
    fig, ax = plt.subplots(nb_rows, nb_columns, figsize=(16, 8))
    plt.suptitle("\nRandom MNIST samples of each digit", fontsize=15)
    
    np.random.seed(seed)
    
    for image_index in range(nb_classes):
        random_data_type = data_types[np.random.randint(0, len(data_types))]
        if random_data_type == "train":
            # raw "train" data
            data   = raw_X_train
            labels = raw_y_train
        else:
            # raw "test" data
            data   = raw_X_test
            labels = raw_y_test
        
        possible_image_indices = np.where(labels == image_index)[0]
        random_image_index = possible_image_indices[np.random.randint(0, possible_image_indices.size)]
        random_image = data[random_image_index]
        
        row_index = image_index // nb_columns
        column_index = image_index % nb_columns
        ax[row_index, column_index].imshow(random_image, cmap="gray")
    
    # resetting the seed
    np.random.seed(None)
    
    plt.subplots_adjust(wspace=0.5, top=0.85)
    
    plt.show()


##############################################################################


def format_raw_MNIST_dataset(
        raw_X_train,
        raw_y_train,
        raw_X_test,
        raw_y_test,
        nb_train_samples,
        nb_val_samples,
        nb_test_samples,
        selected_classes="all",
        nb_shuffles=20,
        seed=None,
        verbose=False
    ):
    """
    Formats the raw MNIST data so that it can be directly interpreted by a
    regular MLP neural network. By design, the arguments `raw_X_train`,
    `raw_y_train`, `raw_X_test` and `raw_y_test` are meant to be the outputs
    of the `load_raw_MNIST_dataset` function of this script
    
    The kwarg `selected_classes` can either be :
        - the string "all" (if you want to work with all the digits ranging
          from 0 to 9)
        - a list/tuple/1D-array containing the specific digits you want to work
          with (e.g. [2, 4, 7])
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of all the args/kwargs, except for `selected_classes`
    
    _validate_raw_MNIST_data(
        raw_X_train,
        raw_y_train,
        raw_X_test,
        raw_y_test
    )
    
    # these 3 arguments will be re-checked a bit later in this function
    assert isinstance(nb_train_samples, int)
    assert isinstance(nb_val_samples,   int)
    assert isinstance(nb_test_samples,  int)
    
    assert isinstance(nb_shuffles, int)
    assert nb_shuffles >= 0
    
    assert isinstance(seed, (type(None), int))
    if isinstance(seed, int):
        assert seed >= 0
    
    assert isinstance(verbose, bool)
    
    # ---------------------------------------------------------------------- #
    
    t_beginning_formatting = time()
    
    # ---------------------------------------------------------------------- #
    
    # Only keeping the selected classes in the raw MNIST data (and checking
    # the validity of the `selected_classes` kwarg)
    
    nb_classes = np.unique(raw_y_train).size # = 10
    
    if isinstance(selected_classes, str):
        selected_classes = selected_classes.lower()
        if selected_classes != "all":
            raise ValueError(f"format_raw_MNIST_dataset (mnist_dataset.py) - Invalid value for the kwarg `selected_classes` : \"{selected_classes}\" (expected : \"all\", or the specific list of selected classes)")
    else:
        if not(isinstance(selected_classes, (list, tuple, np.ndarray))):
            raise TypeError(f"format_raw_MNIST_dataset (mnist_dataset.py) - The `selected_classes` kwarg isn't a string, list, tuple or a 1D numpy array, it's a \"{selected_classes.__class__.__name__}\" !")
        
        if not(isinstance(selected_classes, np.ndarray)):
            selected_classes = np.array(selected_classes)
        check_if_label_vector_is_valid(selected_classes)
        
        distinct_selected_classes = np.unique(selected_classes)
        assert distinct_selected_classes.size >= 2, "format_raw_MNIST_dataset (mnist_dataset.py) - Please select at least 2 distinct classes in the `selected_classes` kwarg !"
        assert np.max(distinct_selected_classes) < nb_classes
        
        indices_of_selected_train_classes = []
        indices_of_selected_test_classes  = []
        
        for selected_class in distinct_selected_classes:
            indices_of_selected_train_class = np.where(raw_y_train == selected_class)[0]
            indices_of_selected_train_classes.append(indices_of_selected_train_class)
            
            indices_of_selected_test_class = np.where(raw_y_test == selected_class)[0]
            indices_of_selected_test_classes.append(indices_of_selected_test_class)
        
        indices_of_selected_train_classes = np.hstack(tuple(indices_of_selected_train_classes))
        indices_of_selected_test_classes  = np.hstack(tuple(indices_of_selected_test_classes))
        
        # actually retrieving the selected classes from the raw data
        raw_X_train = raw_X_train[indices_of_selected_train_classes, :]
        raw_y_train = raw_y_train[indices_of_selected_train_classes]
        raw_X_test  = raw_X_test[indices_of_selected_test_classes, :]
        raw_y_test  = raw_y_test[indices_of_selected_test_classes]
        
        assert np.allclose(np.unique(raw_y_train), distinct_selected_classes)
        assert np.allclose(np.unique(raw_y_test),  distinct_selected_classes)
        
        # updating the number of classes in the raw data
        nb_classes = distinct_selected_classes.size
        
        print(f"\nNB : With the selected classes {list_to_string(distinct_selected_classes)}, here are the new shapes of the raw \"train\" and \"test\" data (from which the formatted data will be extracted) :")
        print(f"    - X_train : {raw_X_train.shape}")
        print(f"    - y_train : {raw_y_train.shape}")
        print(f"    - X_test  : {raw_X_test.shape}")
        print(f"    - y_test  : {raw_y_test.shape}")
    
    # ---------------------------------------------------------------------- #
    
    # Re-checking the validity of the arguments `nb_train_samples`, `nb_val_samples`
    # and `nb_test_samples`
    
    assert nb_train_samples >= nb_classes
    assert nb_val_samples >= nb_classes
    assert nb_test_samples >= nb_classes
    
    # NB : The validation set will be extracted from the raw training set
    total_nb_of_raw_train_samples = raw_X_train.shape[0] # = 60000 (if ALL the classes are selected)
    assert nb_train_samples + nb_val_samples <= total_nb_of_raw_train_samples
    
    total_nb_of_raw_test_samples = raw_X_test.shape[0] # = 10000 (if ALL the classes are selected)
    assert nb_test_samples <= total_nb_of_raw_test_samples
    
    # ---------------------------------------------------------------------- #
    
    # copying the raw data
    X_train = raw_X_train.copy()
    y_train = raw_y_train.copy()
    X_test  = raw_X_test.copy()
    y_test  = raw_y_test.copy()
    
    # flattening the images/samples
    nb_pixels_per_image = raw_X_train.shape[1] * raw_X_train.shape[2] # = 28 * 28 = 784
    X_train = X_train.reshape((total_nb_of_raw_train_samples, nb_pixels_per_image))
    X_test  = X_test.reshape((total_nb_of_raw_test_samples, nb_pixels_per_image))
    
    # casting the "uint8" data into `utils.DEFAULT_DATATYPE` data
    X_train = cast(X_train, utils.DEFAULT_DATATYPE)
    X_test  = cast(X_test,  utils.DEFAULT_DATATYPE)
    
    # casting the "uint8" labels into "int32" labels (optional)
    y_train = np.int_(y_train)
    y_test  = np.int_(y_test)
    
    # normalizing the data between 0 and 1 (by convention)
    normalizing_factor = cast(1, utils.DEFAULT_DATATYPE) / cast(255, utils.DEFAULT_DATATYPE)
    X_train = normalizing_factor * X_train
    X_test  = normalizing_factor * X_test
    
    # getting the formatted "train+val" dataset (from the raw "train" data)
    if total_nb_of_raw_train_samples - (nb_train_samples + nb_val_samples) >= nb_classes:
        X_train_val, _, y_train_val, _ = train_test_split(
            X_train,
            y_train,
            stratify=y_train, # keeping the same class distribution as `y_train`
            train_size=nb_train_samples+nb_val_samples,
            random_state=seed
        )
    else:
        # in this specific case, the `train_test_split` function breaks
        np.random.seed(seed)
        train_val_indices = np.random.choice(np.arange(total_nb_of_raw_train_samples), size=(nb_train_samples + nb_val_samples, ), replace=False)
        np.random.seed(None) # resetting the seed
        X_train_val = X_train[train_val_indices, :].copy()
        y_train_val = y_train[train_val_indices].copy()
    
    assert np.unique(y_train_val).size == nb_classes
    
    # getting the formatted "train" and "val" datasets (directly from the
    # formatted "train+val" dataset)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        stratify=y_train_val, # keeping the same class distribution as `y_train_val` (which had the same class distribution as `y_train`)
        train_size=nb_train_samples,
        random_state=seed
    )
    
    assert np.unique(y_train).size == nb_classes
    assert np.unique(y_val).size == nb_classes
    
    # getting the formatted "test" dataset (from the raw "test" data)
    if total_nb_of_raw_test_samples - nb_test_samples >= nb_classes:
        X_test, _, y_test, _ = train_test_split(
            X_test,
            y_test,
            stratify=y_test, # keeping the same class distribution as `y_test`
            train_size=nb_test_samples,
            random_state=seed
        )
    else:
        # in this specific case, the `train_test_split` function breaks
        np.random.seed(seed)
        test_indices = np.random.choice(np.arange(total_nb_of_raw_test_samples), size=(nb_test_samples, ), replace=False)
        np.random.seed(None) # resetting the seed
        X_test = X_test[test_indices, :].copy()
        y_test = y_test[test_indices].copy()
    
    assert np.unique(y_test).size == nb_classes
    
    # one-hot encoding the label vectors
    y_train = to_categorical(y_train, dtype=utils.DEFAULT_DATATYPE)
    y_val   = to_categorical(y_val,   dtype=utils.DEFAULT_DATATYPE)
    y_test  = to_categorical(y_test,  dtype=utils.DEFAULT_DATATYPE)
    
    # ---------------------------------------------------------------------- #
    
    # shuffling the data
    
    shuffled_indices_train_data      = np.arange(nb_train_samples)
    shuffled_indices_validation_data = np.arange(nb_val_samples)
    shuffled_indices_test_data       = np.arange(nb_test_samples)
    
    np.random.seed(seed)
    for shuffle_index in range(nb_shuffles):
        np.random.shuffle(shuffled_indices_train_data)
        np.random.shuffle(shuffled_indices_validation_data)
        np.random.shuffle(shuffled_indices_test_data)
    np.random.seed(None) # resetting the seed
    
    X_train = X_train[shuffled_indices_train_data, :]
    y_train = y_train[shuffled_indices_train_data, :]
    X_val   = X_val[shuffled_indices_validation_data, :]
    y_val   = y_val[shuffled_indices_validation_data, :]
    X_test  = X_test[shuffled_indices_test_data, :]
    y_test  = y_test[shuffled_indices_test_data, :]
    
    # ---------------------------------------------------------------------- #
    
    # checking the datatype of the final formatted data
    
    check_dtype(X_train, utils.DEFAULT_DATATYPE)
    check_dtype(y_train, utils.DEFAULT_DATATYPE)
    check_dtype(X_val,   utils.DEFAULT_DATATYPE)
    check_dtype(y_val,   utils.DEFAULT_DATATYPE)
    check_dtype(X_test,  utils.DEFAULT_DATATYPE)
    check_dtype(y_test,  utils.DEFAULT_DATATYPE)
    
    # ---------------------------------------------------------------------- #
    
    t_end_formatting = time()
    duration_formatting = t_end_formatting - t_beginning_formatting
    
    if verbose:
        print("\nShapes of the formatted MNIST data :")
        print(f"    - X_train : {X_train.shape}")
        print(f"    - y_train : {y_train.shape}")
        print(f"    - X_val   : {X_val.shape}")
        print(f"    - y_val   : {y_val.shape}")
        print(f"    - X_test  : {X_test.shape}")
        print(f"    - y_test  : {y_test.shape}")
        
        print("\nTypes of the formatted MNIST data :")
        print(f"    - X_train : {get_dtype_of_array(X_train)}")
        print(f"    - y_train : {get_dtype_of_array(y_train)}")
        print(f"    - X_val   : {get_dtype_of_array(X_val)}")
        print(f"    - y_val   : {get_dtype_of_array(y_val)}")
        print(f"    - X_test  : {get_dtype_of_array(X_test)}")
        print(f"    - y_test  : {get_dtype_of_array(y_test)}")
        
        precision_of_printed_info = 3
        print("\nRanges of the formatted MNIST data :")
        print(f"    - X_train : {get_range_of_array(X_train, precision=precision_of_printed_info)} (mean={X_train.mean():.{precision_of_printed_info}f}, std={X_train.std():.{precision_of_printed_info}f})")
        print(f"    - y_train : {get_range_of_array(y_train, precision=precision_of_printed_info)} (one-hot encoded)")
        print(f"    - X_val   : {get_range_of_array(X_val,   precision=precision_of_printed_info)} (mean={X_val.mean():.{precision_of_printed_info}f}, std={X_val.std():.{precision_of_printed_info}f})")
        print(f"    - y_val   : {get_range_of_array(y_val,   precision=precision_of_printed_info)} (one-hot encoded)")
        print(f"    - X_test  : {get_range_of_array(X_test,  precision=precision_of_printed_info)} (mean={X_test.mean():.{precision_of_printed_info}f}, std={X_test.std():.{precision_of_printed_info}f})")
        print(f"    - y_test  : {get_range_of_array(y_test,  precision=precision_of_printed_info)} (one-hot encoded)")
        
        # displaying the class distribution of the final formatted data
        dict_of_label_vectors = {
            "y_train" : categorical_to_vector(y_train),
            "y_val"   : categorical_to_vector(y_val),
            "y_test"  : categorical_to_vector(y_test)
        }
        display_class_distributions(
            dict_of_label_vectors,
            selected_classes=selected_classes,
            precision=2
        )
    
    print(f"\nThe raw MNIST dataset was successfully formatted. Done in {duration_formatting:.3f} seconds")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

