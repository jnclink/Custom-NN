# -*- coding: utf-8 -*-

"""
Functions used to load and format the raw MNIST dataset
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

# only used to load the raw MNIST dataset
from tensorflow.keras.datasets import mnist

# only used to split the data, such that the proportions of the classes in the
# split data are roughly the same as the proportions of the classes in the
# initial raw data (this is done using the VERY USEFUL `stratify` kwarg of the
# `train_test_split` function)
from sklearn.model_selection import train_test_split

from utils import (
    get_type_of_array,
    get_range_of_array,
    to_categorical,
    categorical_to_vector,
    display_distribution_of_classes
)


##############################################################################


def load_raw_MNIST_dataset(verbose=False):
    """
    This function is basically a wrapper of the `mnist.load_data` function
    """
    
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
        print(f"    - X_train : {get_type_of_array(raw_X_train)}")
        print(f"    - y_train : {get_type_of_array(raw_y_train)}")
        print(f"    - X_test  : {get_type_of_array(raw_X_test)}")
        print(f"    - y_test  : {get_type_of_array(raw_y_test)}")
        
        precision = 3
        print("\nRanges of the raw MNIST data (loaded from Keras) :")
        print(f"    - X_train : {get_range_of_array(raw_X_train, precision=precision)}")
        print(f"    - y_train : {get_range_of_array(raw_y_train, precision=precision)}")
        print(f"    - X_test  : {get_range_of_array(raw_X_test, precision=precision)}")
        print(f"    - y_test  : {get_range_of_array(raw_y_test, precision=precision)}")
    
    print(f"\nThe raw MNIST dataset was successfully loaded. Done in {duration_loading:.3f} seconds")
    
    return raw_X_train, raw_y_train, raw_X_test, raw_y_test


def plot_random_images_from_raw_MNIST_dataset(
        raw_X_train,
        raw_y_train,
        raw_X_test,
        raw_y_test,
        seed=None
    ):
    """
    Plots a random sample of each of the 10 digits (from the raw MNIST data)
    """
    
    nb_rows    = 2
    nb_columns = 5
    
    nb_classes = np.unique(raw_y_train).size
    assert nb_rows * nb_columns == nb_classes
    
    data_types = ["train", "test"]
    
    fig, ax = plt.subplots(nb_rows, nb_columns, figsize=(12, 8))
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
    
    plt.show()


def format_raw_MNIST_dataset(
        raw_X_train,
        raw_y_train,
        raw_X_test,
        raw_y_test,
        nb_train_samples,
        nb_val_samples,
        nb_test_samples,
        nb_shuffles=20,
        seed=None,
        verbose=False
    ):
    """
    Formats the raw data so that it can be directly interpreted by a regular
    MLP neural network
    """
    
    # the validation set will be extracted from the raw training set
    total_nb_of_raw_train_samples = raw_X_train.shape[0] # = 60000
    assert nb_train_samples + nb_val_samples <= total_nb_of_raw_train_samples
    
    total_nb_of_raw_test_samples = raw_X_test.shape[0] # = 10000
    assert nb_test_samples <= total_nb_of_raw_test_samples
    
    nb_classes = np.unique(raw_y_train).size # = 10
    assert np.unique(raw_y_test).size == nb_classes
    
    assert nb_train_samples >= nb_classes
    assert nb_val_samples >= nb_classes
    assert nb_test_samples >= nb_classes
    
    t_beginning_formatting = time()
    
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
    
    # converting the "uint8" data into "float32" data
    X_train = X_train.astype("float32")
    X_test  = X_test.astype("float32")
    
    # converting the "uint8" labels into "int32" labels (optional)
    y_train = np.int_(y_train)
    y_test  = np.int_(y_test)
    
    # normalizing the data between 0 and 1 (by convention)
    X_train = (1.0 / 255) * X_train
    X_test  = (1.0 / 255) * X_test
    
    # getting the cropped version of "train+val" (from the raw "train" data)
    if total_nb_of_raw_train_samples - (nb_train_samples + nb_val_samples) >= nb_classes:
        X_train_val, _, y_train_val, _ = train_test_split(
            X_train,
            y_train,
            stratify=y_train, # keeping the same class proportions as `y_train`
            train_size=nb_train_samples+nb_val_samples,
            random_state=seed
        )
    else:
        # in this specific case, the `train_test_split` function breaks
        np.random.seed(seed)
        train_val_indices = np.random.choice(np.arange(total_nb_of_raw_train_samples), size=(nb_train_samples + nb_val_samples, ))
        np.random.seed(None) # resetting the seed
        X_train_val = X_train[train_val_indices, :].copy()
        y_train_val = y_train[train_val_indices].copy()
    
    assert np.unique(y_train_val).size == nb_classes
    
    # getting the "train" and "val" datasets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        stratify=y_train_val, # keeping the same class proportions as `y_train_val`
        train_size=nb_train_samples,
        random_state=seed
    )
    
    assert np.unique(y_train).size == nb_classes
    assert np.unique(y_val).size == nb_classes
    
    # getting the cropped version of "test" (from the raw "test" data)
    if total_nb_of_raw_test_samples - nb_test_samples >= nb_classes:
        X_test, _, y_test, _ = train_test_split(
            X_test,
            y_test,
            stratify=y_test, # keeping the same class proportions as `y_test`
            train_size=nb_test_samples,
            random_state=seed
        )
    else:
        # in this specific case, the `train_test_split` function breaks
        np.random.seed(seed)
        test_indices = np.random.choice(np.arange(total_nb_of_raw_test_samples), size=(nb_test_samples, ))
        np.random.seed(None) # resetting the seed
        X_test = X_test[test_indices, :].copy()
        y_test = y_test[test_indices].copy()
    
    assert np.unique(y_test).size == nb_classes
    
    # one-hot encoding the label vectors
    y_train = to_categorical(y_train, dtype="float32")
    y_val   = to_categorical(y_val,   dtype="float32")
    y_test  = to_categorical(y_test,  dtype="float32")
    
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
        print(f"    - X_train : {get_type_of_array(X_train)}")
        print(f"    - y_train : {get_type_of_array(y_train)}")
        print(f"    - X_val   : {get_type_of_array(X_val)}")
        print(f"    - y_val   : {get_type_of_array(y_val)}")
        print(f"    - X_test  : {get_type_of_array(X_test)}")
        print(f"    - y_test  : {get_type_of_array(y_test)}")
        
        precision = 3
        print("\nRanges of the formatted MNIST data :")
        print(f"    - X_train : {get_range_of_array(X_train, precision=precision)} (mean={X_train.mean():.{precision}f}, std={X_train.std():.{precision}f})")
        print(f"    - y_train : {get_range_of_array(y_train, precision=precision)} (one-hot encoded)")
        print(f"    - X_val   : {get_range_of_array(X_val, precision=precision)} (mean={X_val.mean():.{precision}f}, std={X_val.std():.{precision}f})")
        print(f"    - y_val   : {get_range_of_array(y_val, precision=precision)} (one-hot encoded)")
        print(f"    - X_test  : {get_range_of_array(X_test, precision=precision)} (mean={X_test.mean():.{precision}f}, std={X_test.std():.{precision}f})")
        print(f"    - y_test  : {get_range_of_array(y_test, precision=precision)} (one-hot encoded)")
        
        # displaying the proportions of the digits in the final formatted data
        dict_of_label_vectors = {
            "y_train" : categorical_to_vector(y_train),
            "y_val"   : categorical_to_vector(y_val),
            "y_test"  : categorical_to_vector(y_test)
        }
        display_distribution_of_classes(dict_of_label_vectors)
    
    print(f"\nThe raw MNIST dataset was successfully formatted. Done in {duration_formatting:.3f} seconds")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

