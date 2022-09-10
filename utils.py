# -*- coding: utf-8 -*-

"""
Script defining miscellaneous useful functions
"""

import os
from time import time
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from matplotlib import cm # colormaps
from matplotlib.colors import rgb2hex, Colormap


##############################################################################


# Defining the datatype of ALL the data that will flow through the network
# (i.e. we're defining the "global datatype")

# global variables (these are the 2 only global variables of the entire project)
DEFAULT_DATATYPE = "float32"
DTYPE_RESOLUTION = np.finfo(DEFAULT_DATATYPE).resolution


##############################################################################


def _validate_numpy_datatype(datatype):
    """
    Checks if the specified datatype is a valid NumPy datatype or not
    """
    if isinstance(datatype, str):
        datatype = datatype.lower().replace(" ", "")
    try:
        _ = np.dtype(datatype)
    except:
        raise ValueError(f"_validate_numpy_datatype (utils.py) - Invalid NumPy datatype : \"{str(datatype)}\"")
    return datatype


def _validate_global_datatype(datatype):
    """
    Checks if the specified datatype can be the "global datatype" of the
    network (or not). For now, the only available datatypes are float32
    and float64
    """
    datatype = _validate_numpy_datatype(datatype)
    
    AVAILABLE_DATATYPES = (
        "float32",
        np.float32,
        "float64",
        np.float64
    )
    
    if datatype not in AVAILABLE_DATATYPES:
        if isinstance(datatype, type):
            str_datatype = datatype.__name__
        else:
            str_datatype = str(datatype)
        raise ValueError(f"_validate_global_datatype (utils.py) - Unrecognized value for `datatype` : \"{str_datatype}\" (has to be float32 or float64)")


# checking the hard-coded value of `DEFAULT_DATATYPE` (in fact, this is already
# done by the `np.finfo` method when defining the global variable `DTYPE_RESOLUTION` !)
_validate_global_datatype(DEFAULT_DATATYPE)


def set_global_datatype(datatype):
    """
    Sets the global variable `DEFAULT_DATATYPE` to the specified datatype. The
    resolution of the global datatype (i.e. the global variable `DTYPE_RESOLUTION`)
    is also updated. The resolution is `1e-6` for float32, and `1e-15` for float64
    
    IMPORTANT NOTE
    --------------
    If you ever change the default value of `DEFAULT_DATATYPE` (using this
    function), in order to access the UPDATED version of `DEFAULT_DATATYPE`
    inside the code, make sure you add the following import : `import utils`,
    then use the variable `utils.DEFAULT_DATATYPE`. Please DO NOT add the import
    `from utils import DEFAULT_DATATYPE` before calling this function, as it
    will return the default value of `DEFAULT_DATATYPE`, not the updated one !
    The same goes with the global variable `DTYPE_RESOLUTION`
    """
    if datatype == float:
        set_global_datatype(np.float_)
    
    # checking the specified datatype first
    _validate_global_datatype(datatype)
    
    global DEFAULT_DATATYPE, DTYPE_RESOLUTION
    DEFAULT_DATATYPE = datatype
    DTYPE_RESOLUTION = np.finfo(DEFAULT_DATATYPE).resolution


##############################################################################


# Functions related to the checking/casting of the global datatype of the network


def check_dtype(x, dtype):
    """
    Checks if the datatype of `x` is `dtype` or not. Here, `x` can either be
    a scalar or a vector/matrix. By design, in most cases, `dtype` will be
    equal to `utils.DEFAULT_DATATYPE`
    """
    assert np.isscalar(x) or isinstance(x, np.ndarray)
    dtype = _validate_numpy_datatype(dtype)
    
    if np.isscalar(x):
        assert type(x) == np.dtype(dtype).type
    else:
        # here, `x` is a vector/matrix
        assert x.dtype == dtype


def cast(x, dtype):
    """
    Returns the cast version of `x` to `dtype`. Here, `x` can either be a
    scalar or a vector/matrix. By design, in most cases, `dtype` will be
    equal to `utils.DEFAULT_DATATYPE`
    """
    assert np.isscalar(x) or isinstance(x, np.ndarray)
    dtype = _validate_numpy_datatype(dtype)
    
    if np.isscalar(x):
        cast_x = np.dtype(dtype).type(x)
    else:
        # here, `x` is a vector/matrix
        cast_x = x.astype(dtype)
    
    check_dtype(cast_x, dtype)
    return cast_x


##############################################################################


# Generally useful functions


def list_to_string(L):
    """
    Converts the specified non-empty list (or tuple or 1D numpy array) into
    a string
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the input `L`
    
    if not(isinstance(L, (list, tuple, np.ndarray))):
        raise TypeError(f"list_to_string (utils.py) - The input `L` isn't a list (or a tuple or a 1D numpy array), it's a \"{L.__class__.__name__}\" !")
    
    if isinstance(L, np.ndarray):
        assert len(L.shape) == 1, "list_to_string (utils.py) - The input `L` has to be 1-dimensional !"
    
    nb_elements = len(L)
    assert nb_elements > 0, "list_to_string (utils.py) - The input `L` is empty !"
    
    # ---------------------------------------------------------------------- #
    
    str_L = ""
    
    for element_index, element in enumerate(L):
        if isinstance(element, type):
            str_element = element.__name__
        else:
            str_element = str(element)
        str_element = f"\"{str_element}\""
        
        if element_index == 0:
            # first element of the list/tuple
            str_L += str_element
        elif element_index < nb_elements - 1:
            str_L += ", " + str_element
        else:
            # last element of the list/tuple
            str_L += " and " + str_element
    
    return str_L


def count_nb_decimals_places(x, max_precision=6):
    """
    Returns the number of decimal places of the scalar `x`
    """
    assert np.isscalar(x)
    
    assert isinstance(max_precision, int)
    assert max_precision >= 0
    
    # converting `x` into a positive number, since it doesn't affect the number
    # of decimal places
    x = round(float(abs(x)), max_precision)
    
    if x == int(x):
        # here, `x` is a positive integer
        return 0
    elif x < 1:
        # here, `x` is a float in the range ]0, 1[
        return len(str(x)) - 2
    else:
        # here, `x` is a non-integer float in the range ]1, infinity[, therefore
        # `x - int(x)` will be a float in the range ]0, 1[
        return len(str(x - int(x))) - 2


def clear_currently_printed_row(max_size_of_row=150):
    """
    Clears the currently printed row, and sets the pointer of the `print`
    function at the very beginning of that same row
    """
    assert isinstance(max_size_of_row, int)
    assert max_size_of_row >= 0
    
    blank_row = " " * max_size_of_row
    print(blank_row, end="\r")


def progress_bar(
        current_index,
        total_nb_elements,
        progress_bar_size=15
    ):
    """
    Returns a progress bar (as a string) corresponding to the progress of
    `current_index` relative to `total_nb_elements`. The value of the kwarg
    `progress_bar_size` indicates how long the content of the progress bar is
    
    For example, `progress_bar(70, 100, progress_bar_size=10)` will return
    the following string : "[======>...]"
    """
    # ---------------------------------------------------------------------- #
    
    # checking the specified arguments
    
    assert isinstance(current_index, int)
    assert current_index >= 1
    
    assert isinstance(total_nb_elements, int)
    assert total_nb_elements >= max(1, current_index)
    
    assert isinstance(progress_bar_size, int)
    assert progress_bar_size >= 2
    
    # ---------------------------------------------------------------------- #
    
    if current_index == total_nb_elements:
        str_progress_bar = "[" + "=" * progress_bar_size + "]"
    else:
        progress_ratio = float(current_index) / total_nb_elements
        nb_equal_signs = max(1, int(round(progress_ratio * progress_bar_size))) - 1
        
        nb_dots = progress_bar_size - nb_equal_signs - 1
        
        str_progress_bar = "[" + "=" * nb_equal_signs + ">" + "." * nb_dots + "]"
    
    assert len(str_progress_bar) == progress_bar_size + 2
    
    return str_progress_bar


##############################################################################


# Other functions used to validate values/objects/arguments


def _validate_raw_MNIST_dataset(
        raw_X_train,
        raw_y_train,
        raw_X_test,
        raw_y_test
    ):
    """
    Checks if the specified raw MNIST data is valid or not. By design, the
    arguments `raw_X_train`, `raw_y_train`, `raw_X_test` and `raw_y_test` are
    meant to be the outputs of the `load_raw_MNIST_dataset_from_disk` function
    of the "mnist_dataset.py" script
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
    
    NB_CLASSES = 10
    expected_classes = np.arange(NB_CLASSES)
    assert np.allclose(np.unique(raw_y_train), expected_classes)
    assert np.allclose(np.unique(raw_y_test),  expected_classes)


def _validate_label_vector(y, max_class=9, is_whole_label_vector=True):
    """
    Checks the validity of the label vector `y`
    """
    assert isinstance(max_class, int)
    assert max_class >= 1
    
    assert isinstance(is_whole_label_vector, bool)
    
    assert isinstance(y, np.ndarray)
    assert len(y.shape) == 1
    assert issubclass(y.dtype.type, np.integer)
    
    assert np.min(y) >= 0
    assert np.max(y) <= max_class
    
    if is_whole_label_vector:
        nb_classes = np.unique(y).size
        assert nb_classes >= 2


def _validate_selected_classes(selected_classes):
    """
    Checks the validity of the `selected_classes` argument, and returns the
    potentially corrected version of `selected_classes`
    
    Here, `selected_classes` can either be :
        - the string "all" (if you're working with all the digits ranging
          from 0 to 9)
        - a list/tuple/1D-array containing the specific digits you're
          working with (e.g. [2, 4, 7])
    """
    if isinstance(selected_classes, str):
        selected_classes = selected_classes.lower()
        if selected_classes != "all":
            raise ValueError(f"_validate_selected_classes (utils.py) - Invalid value for the argument `selected_classes` : \"{selected_classes}\" (expected : \"all\", or the specific list of selected classes)")
    else:
        if not(isinstance(selected_classes, (list, tuple, np.ndarray))):
            raise TypeError(f"_validate_selected_classes (utils.py) - The `selected_classes` argument isn't a string, list, tuple or a 1D numpy array, it's a \"{selected_classes.__class__.__name__}\" !")
        
        if not(isinstance(selected_classes, np.ndarray)):
            selected_classes = np.array(selected_classes)
        _validate_label_vector(selected_classes)
        
        distinct_selected_classes = np.unique(selected_classes)
        assert distinct_selected_classes.size >= 2, "_validate_selected_classes (utils.py) - Please select at least 2 distinct classes in the `selected_classes` argument !"
        
        selected_classes = distinct_selected_classes
    
    return selected_classes


def _validate_loss_inputs(y_true, y_pred):
    """
    Checks if `y_true` and `y_pred` are valid or not (as inputs of a
    loss function)
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) in [1, 2]
    
    global DEFAULT_DATATYPE
    check_dtype(y_true, DEFAULT_DATATYPE)
    check_dtype(y_pred, DEFAULT_DATATYPE)


def _validate_activation_input(x, can_be_scalar=True):
    """
    Checks the validity of `x` (as an input of an activation function)
    
    If `can_be_scalar` is set to `True`, `x` can either be a scalar, a 1D
    vector or a 2D matrix (usually the latter). Otherwise, `x` can either be
    a 1D vector or a 2D matrix
    """
    assert isinstance(can_be_scalar, bool)
    
    if can_be_scalar:
        assert np.isscalar(x) or isinstance(x, np.ndarray)
    else:
        assert isinstance(x, np.ndarray)
    
    if isinstance(x, np.ndarray):
        assert len(x.shape) in [1, 2]
    
    global DEFAULT_DATATYPE
    check_dtype(x, DEFAULT_DATATYPE)


def _validate_leaky_ReLU_coeff(leaky_ReLU_coeff):
    """
    Checks if `leaky_ReLU_coeff` is valid or not
    """
    assert isinstance(leaky_ReLU_coeff, float)
    
    global DEFAULT_DATATYPE
    leaky_ReLU_coeff = cast(leaky_ReLU_coeff, DEFAULT_DATATYPE)
    
    assert (leaky_ReLU_coeff > 0) and (leaky_ReLU_coeff < 1)
    
    return leaky_ReLU_coeff


##############################################################################


# Functions related to the `mnist_dataset.py` script


def _download_progress_bar(
        block_index,
        block_size_in_bytes,
        total_size_of_data_in_bytes
    ):
    """
    Prints the progress bar concerning the download of the raw MNIST data
    
    The signature of this function is imposed by the `urlretrieve` method
    of the `urllib.request` module (cf. the `reporthook` kwarg of the
    `urlretrieve` method)
    """
    # ---------------------------------------------------------------------- #
    
    # checking the specified arguments
    
    assert isinstance(block_index, int)
    assert block_index >= 0
    
    if block_index == 0:
        # when `block_index` is equal to zero, it's to signal that the
        # downloading process has just begun, but no data has actually
        # been retrieved yet
        return
    
    assert isinstance(block_size_in_bytes, int)
    assert block_size_in_bytes > 0
    
    assert isinstance(total_size_of_data_in_bytes, int)
    assert total_size_of_data_in_bytes > 0
    
    # this formula assumes that all the blocks are the same size (except for
    # maybe the very last one, if `block_size_in_bytes` doesn't divide
    # `total_size_of_data_in_bytes`), which is the case in practice
    total_nb_blocks = (total_size_of_data_in_bytes + block_size_in_bytes - 1) // block_size_in_bytes
    
    assert block_index <= total_nb_blocks
    
    # ---------------------------------------------------------------------- #
    
    # defining the number of times the progress bar will be updated during
    # the entire download
    nb_progress_bar_updates = 10
    
    block_index_update_step = total_nb_blocks // nb_progress_bar_updates
    
    if (block_index % block_index_update_step == 0) or (block_index in [1, total_nb_blocks]):
        size_of_already_downloaded_data_in_bytes = block_index * block_size_in_bytes
        
        if block_index == total_nb_blocks:
            assert size_of_already_downloaded_data_in_bytes >= total_size_of_data_in_bytes
            size_of_already_downloaded_data_in_bytes = total_size_of_data_in_bytes
        
        current_progress_bar = progress_bar(
            size_of_already_downloaded_data_in_bytes,
            total_size_of_data_in_bytes,
            progress_bar_size=50 # by default
        )
        
        # by definition of a megabyte
        nb_bytes_in_one_megabyte = 1024**2
        
        # converting the sizes from bytes to megabytes (for display purposes only)
        size_of_already_downloaded_data_in_megabytes = float(size_of_already_downloaded_data_in_bytes) / nb_bytes_in_one_megabyte
        total_size_of_data_in_megabytes = float(total_size_of_data_in_bytes) / nb_bytes_in_one_megabyte
        
        # by default
        precision_sizes = 2
        
        str_size_of_already_downloaded_data_in_megabytes = f"{size_of_already_downloaded_data_in_megabytes:.{precision_sizes}f}"
        if size_of_already_downloaded_data_in_megabytes < 10:
            str_size_of_already_downloaded_data_in_megabytes = "0" + str_size_of_already_downloaded_data_in_megabytes
        str_total_size_of_data_in_megabytes = f"{total_size_of_data_in_megabytes:.{precision_sizes}f}"
        
        current_progress_bar += f" Downloaded {str_size_of_already_downloaded_data_in_megabytes}/{str_total_size_of_data_in_megabytes} MB"
        
        clear_currently_printed_row()
        print(current_progress_bar, end="\r")


def _download_raw_MNIST_dataset():
    r"""
    Automatically downloads the raw MNIST data (as a single file), and saves
    it to the following location on your disk :
        - on Windows : "C:\Users\YourUsername\.Custom-MLP\datasets\MNIST\raw_MNIST_data.npz"
        - on Linux   : "/home/YourUsername/.Custom-MLP/datasets/MNIST/raw_MNIST_data.npz"
    
    Naturally, if this is the very first time you call this function, you'll
    need to have an internet connection !
    
    The downloaded file has a size of about 11 MB, and is retrieved from the
    following URL :
    https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    
    If the data has already been downloaded, it's simply retrieved from your disk
    
    Also, the absolute path of the downloaded data is returned
    
    Sidenote
    --------
    This function is basically equivalent to the `tensorflow.keras.datasets.mnist.load_data`
    method. The only real difference is that the downloaded data has a different
    location on your disk. The reason as for why the previous `mnist.load_data`
    method isn't used is simply because we do NOT want to import TensorFlow !
    Indeed, in my opinion, it would be kind of awkward to import TensorFlow in
    a project aiming to implement a Deep Learning model *from scratch* !
    Note that, if you use the `mnist.load_data` method of the TensorFlow module,
    the raw MNIST data will be saved at the following location on your disk :
        - on Windows : "C:\Users\YourUsername\.keras\datasets\mnist.npz"
        - on Linux   : "/home/YourUsername/.keras/datasets/mnist.npz"
    """
    
    # creating the folder that will contain the raw MNIST data (if it doesn't
    # already exist)
    default_data_directory = os.path.join(
        os.path.expanduser("~"),
        ".Custom-MLP",
        "datasets",
        "MNIST"
    )
    os.makedirs(default_data_directory, exist_ok=True)
    
    # here, `default_data_filename` has to be a "*.npz" filename
    default_data_filename = "raw_MNIST_data.npz"
    assert default_data_filename[-4 : ] == ".npz"
    
    # defining the absolute path of the downloaded file
    default_path_of_downloaded_data = os.path.join(
        default_data_directory,
        default_data_filename
    )
    
    # if the data already exists on your disk, there is no need to re-download it
    download_is_required = not(os.path.exists(default_path_of_downloaded_data))
    
    if download_is_required:
        try:
            data_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
            
            print(f"\nDownloading the raw MNIST data from the URL \"{data_URL}\". This might take a couple of seconds ...\n")
            t_beginning_downloading = time()
            
            # actually downloading the raw MNIST data from `data_URL`, and
            # saving it to the location `default_path_of_downloaded_data`
            urlretrieve(
                url=data_URL,
                filename=default_path_of_downloaded_data,
                reporthook=_download_progress_bar
            )
            
            assert os.path.exists(default_path_of_downloaded_data)
            
            t_end_downloading = time()
            duration_downloading = t_end_downloading - t_beginning_downloading
            print(f"\n\nSuccessfully downloaded the raw MNIST data to the location \"{default_path_of_downloaded_data}\". Done in {duration_downloading:.3f} seconds")
        except (Exception, KeyboardInterrupt):
            print("\n")
            if os.path.exists(default_path_of_downloaded_data):
                os.remove(default_path_of_downloaded_data)
            raise
    
    return default_path_of_downloaded_data


def get_dtype_of_array(array):
    """
    Returns the datatype of an array (as a string)
    """
    assert isinstance(array, np.ndarray)
    str_dtype = "numpy." + str(array.dtype)
    return str_dtype


def get_range_of_array(array, precision=3):
    """
    Returns the range of an array (as a string)
    """
    assert isinstance(array, np.ndarray)
    
    assert isinstance(precision, int)
    assert precision >= 0
    
    min_element_of_array = np.min(array)
    max_element_of_array = np.max(array)
    
    FLOAT_DATATYPES = (
        np.float16,
        np.float32,
        np.float64
    )
    
    INTEGER_DATATYPES = (
        np.int8,  np.uint8,
        np.int16, np.uint16,
        np.int32, np.uint32,
        np.int64, np.uint64
    )
    
    array_dtype = array.dtype.type
    
    if array_dtype in FLOAT_DATATYPES:
        # floating-point data
        str_range = f"{min_element_of_array:.{precision}f} -> {max_element_of_array:.{precision}f}"
    elif array_dtype in INTEGER_DATATYPES:
        # integer data
        str_range = f"{min_element_of_array} -> {max_element_of_array}"
    else:
        raise TypeError(f"get_range_of_array (utils.py) - The input `array` has an unrecognized datatype : \"{array_dtype.__name__}\" (it has to be numerical, i.e. float or int)")
    
    return str_range


def display_class_distributions(
        dict_of_label_vectors,
        selected_classes="all",
        precision=2
    ):
    """
    Prints the class distributions of the specified label vectors (i.e. the
    values of the dictionary `dict_of_label_vectors`). The keys of `dict_of_label_vectors`
    are the names of the corresponding label vectors (as strings)
    
    The kwarg `selected_classes` can either be :
        - the string "all" (if you're working with all the digits ranging
          from 0 to 9)
        - a list/tuple/1D-array containing the specific digits you're
          working with (e.g. [2, 4, 7])
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    assert isinstance(dict_of_label_vectors, dict)
    
    selected_classes = _validate_selected_classes(selected_classes)
    
    assert isinstance(precision, int)
    assert precision >= 0
    
    # ---------------------------------------------------------------------- #
    
    displayed_string = "\nClass distributions :\n"
    
    for label_vector_name, label_vector in dict_of_label_vectors.items():
        # checking the validity of the individual components of `dict_of_label_vectors`
        assert isinstance(label_vector_name, str)
        _validate_label_vector(label_vector)
        
        displayed_string += f"\n{label_vector_name} :"
        
        nb_classes = np.unique(label_vector).size
        assert nb_classes >= 2
        
        if isinstance(selected_classes, str):
            # here, `selected_classes` is equal to the string "all"
            classes = np.arange(nb_classes)
        else:
            assert nb_classes == selected_classes.size
            classes = selected_classes
        
        nb_labels = label_vector.size
        
        for digit_index, digit in enumerate(classes):
            nb_corresponding_digits = np.where(label_vector == digit_index)[0].size
            proportion = 100 * float(nb_corresponding_digits) / nb_labels
            str_proportion = f"{proportion:.{precision}f}"
            if proportion < 10:
                str_proportion = "0" + str_proportion
            displayed_string += f"\n    {digit} --> {str_proportion} %"
    
    print(displayed_string)


##############################################################################


# Functions used to switch from an integer vector of labels to a one-hot encoded
# matrix (and vice-versa)


def vector_to_categorical(y, dtype=int):
    """
    Performs one-hot encoding on a 1D vector of INTEGER labels
    """
    _validate_label_vector(y)
    dtype = _validate_numpy_datatype(dtype)
    
    nb_labels = y.size
    
    distinct_labels, distinct_labels_inverse = np.unique(y, return_inverse=True)
    nb_classes = distinct_labels.size
    assert nb_classes >= 2
    
    y_categorical = np.zeros((nb_labels, nb_classes), dtype=dtype)
    
    for label_index, label in enumerate(distinct_labels_inverse):
        # by definition
        y_categorical[label_index, label] = 1
    
    return y_categorical


def categorical_to_vector(y_categorical):
    """
    Converts a 2D categorical matrix (one-hot encoded matrix or logits) into
    its associated 1D vector of INTEGER labels
    """
    # checking the validity of `y_categorical`
    assert isinstance(y_categorical, np.ndarray)
    assert len(y_categorical.shape) == 2
    global DEFAULT_DATATYPE
    check_dtype(y_categorical, DEFAULT_DATATYPE)
    
    # by definition
    y = np.argmax(y_categorical, axis=1)
    
    _validate_label_vector(y, is_whole_label_vector=False)
    return y


##############################################################################


# Main function used to split the input data into batches


def split_data_into_batches(
        data,
        batch_size,
        labels=None,
        normalize_batches=True,
        nb_shuffles=10,
        seed=None
    ):
    """
    Splits the input data and labels into batches with `batch_size` samples each
    (if `batch_size` doesn't divide the number of samples, then the very last
    batch will simply have `nb_samples % batch_size` samples)
    
    Here, if `labels` is not equal to `None`, it can either be a 1D vector of
    INTEGER labels or its one-hot encoded equivalent (in that case, `labels`
    will be a 2D matrix)
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    # checking the validity of the argument `data`
    assert isinstance(data, np.ndarray)
    assert len(data.shape) == 2
    nb_samples, nb_features_per_sample = data.shape
    assert nb_features_per_sample >= 2
    
    assert isinstance(batch_size, int)
    assert (batch_size >= 1) and (batch_size <= nb_samples)
    
    # checking the validity of the `labels` kwarg
    assert isinstance(labels, (type(None), np.ndarray))
    if labels is not None:
        assert len(labels.shape) in [1, 2]
        if len(labels.shape) == 1:
            _validate_label_vector(labels)
        elif len(labels.shape) == 2:
            nb_classes = labels.shape[1]
            assert nb_classes >= 2
        assert labels.shape[0] == nb_samples
    
    assert isinstance(normalize_batches, bool)
    
    assert isinstance(nb_shuffles, int)
    assert nb_shuffles >= 0
    
    if nb_shuffles > 0:
        assert isinstance(seed, (type(None), int))
        if isinstance(seed, int):
            assert seed >= 0
    
    # ---------------------------------------------------------------------- #
    
    # initialization
    
    batches = {
        "data" : []
    }
    
    if labels is not None:
        batches["labels"] = []
    
    batch_indices = np.arange(nb_samples)
    
    if nb_shuffles > 0:
        # shuffling the batch indices
        np.random.seed(seed)
        for shuffle_index in range(nb_shuffles):
            np.random.shuffle(batch_indices)
        np.random.seed(None) # resetting the seed
    
    if normalize_batches:
        # here we're assuming that each sample does NOT have a standard
        # deviation equal to zero (i.e. we're assuming that there are at
        # least 2 different pixel values in each sample)
        used_data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
    else:
        used_data = data.copy()
    
    # ---------------------------------------------------------------------- #
    
    # actually splitting the data and labels into batches
    
    for first_index_of_batch in range(0, nb_samples, batch_size):
        last_index_of_batch = first_index_of_batch + batch_size
        
        indices_of_current_batch = batch_indices[first_index_of_batch : last_index_of_batch]
        
        # checking if the batch size is correct
        if indices_of_current_batch.size != batch_size:
            # if the batch size isn't equal to `batch_size`, then it means
            # that we are generating the very last batch (it also implies that
            # `batch_size` doesn't divide `nb_samples`)
            assert first_index_of_batch == nb_samples - nb_samples % batch_size
            assert indices_of_current_batch.size > 0
            assert indices_of_current_batch.size == nb_samples % batch_size
        
        data_current_batch = used_data[indices_of_current_batch, :]
        batches["data"].append(data_current_batch)
        
        if labels is not None:
            labels_current_batch = labels[indices_of_current_batch]
            batches["labels"].append(labels_current_batch)
    
    # ---------------------------------------------------------------------- #
    
    # checking if the resulting batches are valid or not
    
    assert np.vstack(tuple(batches["data"])).shape == data.shape
    
    if labels is not None:
        if len(labels.shape) == 1:
            # in this case, the labels are a 1D vector of INTEGER values
            stacking_function = np.hstack
        else:
            # in this case, the labels are one-hot encoded (2D matrix)
            assert len(labels.shape) == 2
            stacking_function = np.vstack
        assert stacking_function(tuple(batches["labels"])).shape == labels.shape
    
    expected_nb_batches = (nb_samples + batch_size - 1) // batch_size
    assert len(batches["data"]) == expected_nb_batches
    if labels is not None:
        assert len(batches["labels"]) == expected_nb_batches
    
    # ---------------------------------------------------------------------- #
    
    return batches


##############################################################################


# Functions related to the accuracy metric


def accuracy_score(y_true, y_pred, normalize=True):
    """
    Returns the proportion of the correctly predicted samples. The returned
    proportion lies between 0 and 1
    
    Here, `y_true` and `y_pred` are 1D vectors of INTEGER labels
    """
    # checking the validity of `y_true` and `y_pred`
    _validate_label_vector(y_true, is_whole_label_vector=False)
    _validate_label_vector(y_pred, is_whole_label_vector=False)
    assert y_true.size == y_pred.size
    
    assert isinstance(normalize, bool)
    
    acc_score = np.where(y_true == y_pred)[0].size
    if normalize:
        nb_test_samples = y_true.size
        acc_score = float(acc_score) / nb_test_samples
    return acc_score


def confusion_matrix(y_true, y_pred):
    """
    Returns the raw confusion matrix of `y_true` and `y_pred`. Its shape will
    be `(nb_classes, nb_classes)`, and, for all integers `i` and `j` in the
    range [0, nb_classes - 1], the value of `conf_matrix[i, j]` (say, for
    instance, `N`) indicates that :
        - Out of all the test samples that were predicted to belong to class `i`,
          `N` of them actually belonged to class `j`
        - Or, equivalently, out of all the test samples that actually belonged
          to class `j`, `N` of them were predicted to belong to class `i`
    
    Here, `y_true` and `y_pred` are 1D vectors of INTEGER labels
    
    Sidenote
    --------
    If you decide to use the `confusion_matrix` function of the `sklearn.metrics`
    module, just be aware that the output of their function is the TRANSPOSED
    version of the "common" definition of the confusion matrix (i.e. the
    transposed version of the output of this function)
    """
    # checking the validity of `y_true` and `y_pred`
    _validate_label_vector(y_true)
    _validate_label_vector(y_pred)
    assert y_true.size == y_pred.size
    
    nb_classes = np.unique(y_true).size
    assert nb_classes >= 2
    
    conf_matrix = np.zeros((nb_classes, nb_classes), dtype=int)
    
    for predicted_class, actual_class in zip(y_pred, y_true):
        # by definition (the rows are the predicted classes and the columns
        # are the true classes)
        conf_matrix[predicted_class, actual_class] += 1
    
    return conf_matrix


##############################################################################


# Functions related to the display of the confusion matrix


def highlight_diagonal(dataframe, background_color):
    """
    Function that is only used by the `print_confusion_matrix` function
    (of this script). Returns a dataframe mask with the CSS property
    "background-color" in all of its diagonal elements, and an empty string
    everywhere else
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe.shape) == 2
    
    assert isinstance(background_color, str)
    assert len(background_color) > 0
    assert " " not in background_color
    background_color = background_color.lower()
    
    # ---------------------------------------------------------------------- #
    
    diagonal_mask = np.full(dataframe.shape, "", dtype="<U24")
    
    CSS_property = f"background-color: {background_color}"
    np.fill_diagonal(diagonal_mask, CSS_property)
    
    diagonal_mask_as_dataframe = pd.DataFrame(
        diagonal_mask,
        index=dataframe.index,
        columns=dataframe.columns
    )
    
    return diagonal_mask_as_dataframe


def highlight_all_cells(value, colormap):
    """
    Function that is only used by the `print_confusion_matrix` function
    (of this script). Returns a CSS property "background-color", where
    the intensity of the associated color (relative to the specified colormap)
    is proportional to `value`
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    assert isinstance(value, (str, float, np.float_))
    if isinstance(value, str):
        value = float(value[ : -2])
    assert (value >= 0) and (value <= 100)
    
    assert issubclass(type(colormap), Colormap)
    
    # ---------------------------------------------------------------------- #
    
    # this scaling factor scales down the maximum intensity of the color
    scaling_factor = 0.75
    assert (scaling_factor > 0) and (scaling_factor <= 1)
    
    color_intensity = scaling_factor * (value / 100) # has to lie in the range [0, 1]
    hex_color_of_cell = rgb2hex(colormap(color_intensity))
    
    CSS_property = f"background-color: {hex_color_of_cell}"
    
    return CSS_property


def print_confusion_matrix(
        conf_matrix,
        selected_classes="all",
        normalize="rows",
        precision=1,
        initial_spacing=1,
        display_with_line_breaks=True,
        jupyter_notebook=False,
        color="green"
    ):
    """
    Prints the confusion matrix in a more user-friendly way
    
    Here, `conf_matrix` is a non-normalized confusion matrix (i.e. a raw,
    integer-valued confusion matrix)
    
    The kwarg `selected_classes` can either be :
        - the string "all" (if you're working with all the digits ranging
          from 0 to 9)
        - a list/tuple/1D-array containing the specific digits you're
          working with (e.g. [2, 4, 7])
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    assert isinstance(conf_matrix, np.ndarray)
    assert issubclass(conf_matrix.dtype.type, np.integer)
    assert len(conf_matrix.shape) == 2
    nb_classes = conf_matrix.shape[0]
    assert nb_classes >= 2
    assert conf_matrix.shape[1] == nb_classes
    
    selected_classes = _validate_selected_classes(selected_classes)
    if isinstance(selected_classes, str):
        # here, `selected_classes` is equal to the string "all"
        class_names = [str(digit) for digit in range(nb_classes)]
    else:
        class_names = [str(digit) for digit in selected_classes]
    
    assert isinstance(normalize, str)
    normalize = normalize.lower()
    possible_values_for_normalize_kwarg = ["rows", "columns", "no"]
    if normalize not in possible_values_for_normalize_kwarg:
        raise ValueError(f"get_confusion_matrix_as_dataframe (utils.py) - Unrecognized value for the `normalize` kwarg : \"{normalize}\" (possible values : {list_to_string(possible_values_for_normalize_kwarg)})")
    
    if normalize != "no":
        assert isinstance(precision, int)
        assert precision >= 0
    
    assert isinstance(jupyter_notebook, bool)
    
    if not(jupyter_notebook):
        # the `initial_spacing` kwarg will not be used if `jupyter_notebook`
        # is set to `True`
        assert isinstance(initial_spacing, int)
        assert initial_spacing >= 0
        
        if normalize != "no":
            assert isinstance(display_with_line_breaks, bool)
            pd.set_option("expand_frame_repr", display_with_line_breaks)
    else:
        assert isinstance(color, str)
        assert len(color) > 0
        assert " " not in color
        color = color.lower()
        
        # keys   : generic color names
        # values : (associated CSS/HTML color names, associated Matplotlib colormaps)
        COLORS_AND_COLORMAPS = {
            "green"  : ("green",     cm.Greens),
            "blue"   : ("blue",      cm.Blues),
            "purple" : ("indigo",    cm.Purples),
            "red"    : ("red",       cm.Reds),
            "orange" : ("orangered", cm.Oranges)
        }
        
        if color not in COLORS_AND_COLORMAPS:
            raise ValueError(f"print_confusion_matrix (utils.py) - Unrecognized value for the `color` kwarg : \"{color}\" (possible color names : {list_to_string(list(COLORS_AND_COLORMAPS.keys()))})")
    
    # ---------------------------------------------------------------------- #
    
    # setting some options of the Pandas module (for convenience purposes)
    
    pd.options.mode.chained_assignment = None
    
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    
    # ---------------------------------------------------------------------- #
    
    # converting the confusion matrix into its associated dataframe
    
    if normalize != "no":
        normalized_conf_matrix = np.float_(np.copy(conf_matrix))
        
        for class_index in range(nb_classes):
            if normalize == "rows":
                # computing the PRECISION of the (predicted) class at row index `class_index`
                sum_of_row = np.sum(normalized_conf_matrix[class_index, :])
                if np.allclose(sum_of_row, 0.0):
                    normalized_conf_matrix[class_index, :] = 0
                else:
                    normalized_conf_matrix[class_index, :] /= sum_of_row
            
            elif normalize == "columns":
                # computing the RECALL of the (actual) class at column index `class_index`
                sum_of_column = np.sum(normalized_conf_matrix[:, class_index])
                if np.allclose(sum_of_column, 0.0):
                    associated_class = class_names[class_index]
                    raise Exception(f"print_confusion_matrix (utils.py) - The true class \"{associated_class}\" (class_index={class_index}) isn't represented in the confusion matrix !")
                normalized_conf_matrix[:, class_index] /= sum_of_column
        
        normalized_conf_matrix = np.round(100 * normalized_conf_matrix, precision)
        
        conf_matrix_as_dataframe = pd.DataFrame(
            normalized_conf_matrix,
            index=class_names,
            columns=class_names
        )
        
        for class_index, class_name in enumerate(class_names):
            # adding the "%" symbol to all the elements of the dataframe
            conf_matrix_as_dataframe[class_name] = conf_matrix_as_dataframe[class_name].astype(str) + " %"
            
            if not(jupyter_notebook):
                # highlighting the diagonal values with vertical bars
                diagonal_percentage = conf_matrix_as_dataframe[class_name][class_index]
                conf_matrix_as_dataframe[class_name][class_index] = f"|| {diagonal_percentage} ||"
    else:
        # here, the `normalize` kwarg is equal to "no", therefore the raw
        # confusion matrix will be printed
        conf_matrix_as_dataframe = pd.DataFrame(
            conf_matrix,
            index=class_names,
            columns=class_names
        )
    
    # by definition
    conf_matrix_as_dataframe.columns.name = "ACTUAL"
    conf_matrix_as_dataframe.index.name   = "PREDICTED"
    
    print(f"\nCONFUSION MATRIX (normalized=\"{normalize}\") :")
    
    # ---------------------------------------------------------------------- #
    
    # highlighting the confusion matrix in (shades of) a specific color (on
    # Jupyter notebook only)
    
    if jupyter_notebook:
        if normalize == "no":
            conf_matrix_styler = conf_matrix_as_dataframe.style.apply(
                highlight_diagonal,
                axis=None,
                background_color=COLORS_AND_COLORMAPS[color][0]
            )
        else:
            conf_matrix_styler = conf_matrix_as_dataframe.style.applymap(
                highlight_all_cells,
                colormap=COLORS_AND_COLORMAPS[color][1]
            )
        
        return conf_matrix_styler
    
    # ---------------------------------------------------------------------- #
    
    # adding the initial spacing
    initial_spacing = " " * initial_spacing
    str_conf_matrix_as_dataframe = f"\n{initial_spacing}" + str(conf_matrix_as_dataframe).replace("\n", f"\n{initial_spacing}") + "\n"
    
    print(str_conf_matrix_as_dataframe)

