# -*- coding: utf-8 -*-

"""
Script defining some miscellaneous useful functions
"""

import os
from time import perf_counter
from urllib.request import urlretrieve
from hashlib import sha256

try:
    from IPython.display import display
except:
    # The `IPython.display.display` method will only be called if the
    # `print_confusion_matrix` function (of this script) is run from a
    # Jupyter notebook, which itself requires an IPython backend. Therefore,
    # if the import fails, then that automatically implies that the IPython
    # backend is not available, which in turn implies that the code is not
    # being run on a Jupyter notebook, meaning that the `IPython.display.display`
    # method would have never been called anyway !
    pass

import numpy as np
from pandas import DataFrame, set_option

# imports related to Matplotlib colormaps
from matplotlib import cm
from matplotlib.colors import Colormap, rgb2hex

# used to print colored text in a Python console/terminal
from colorama import init, Back, Style
init() # enables the ability to print colored text in the standard output (of Python consoles/terminals)


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
    will return the default value of `DEFAULT_DATATYPE`, NOT the updated one !
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
    equal to `DEFAULT_DATATYPE`
    """
    # checking the specified arguments
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
    equal to `DEFAULT_DATATYPE`
    """
    # checking the specified arguments
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


def clear_currently_printed_row(max_size_of_row=150):
    """
    Clears the currently printed row, and sets the pointer of the `print`
    function at the very beginning of that same row
    """
    # checking the specified argument
    assert isinstance(max_size_of_row, int)
    assert max_size_of_row >= 1
    
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
    assert total_nb_elements >= current_index
    
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


def standardize_data(data):
    """
    Normalizes a matrix such that its mean is 0 and its standard deviation
    is 1 (i.e. it returns the standardized version of the matrix). By default,
    if `data` is a 2D matrix, then its rows will be standardized
    
    Here, `data` can either be a 1D vector or a 2D matrix
    """
    # checking the validity of `data`
    assert isinstance(data, np.ndarray)
    assert len(data.shape) in [1, 2]
    global DEFAULT_DATATYPE
    check_dtype(data, DEFAULT_DATATYPE)
    
    # checking if there are any samples that have a standard deviation of zero
    # (in that case, it means that the sample is filled with the same constant value)
    data_std = data.std(axis=-1, keepdims=True)
    for sample_std in data_std.flatten():
        assert not(np.allclose(sample_std, 0.0))
    
    # actually standardizing the data
    standardized_data = (data - data.mean(axis=-1, keepdims=True)) / data_std
    
    assert standardized_data.shape == data.shape
    check_dtype(standardized_data, DEFAULT_DATATYPE)
    
    return standardized_data


def is_being_run_on_jupyter_notebook():
    """
    Returns a boolean indicating whether the code is currently being run on
    a Jupyter notebook or not
    """
    try:
        current_IPython_shell_name = get_ipython().__class__.__name__
        return current_IPython_shell_name == "ZMQInteractiveShell"
    except (Exception, NameError):
        return False


def count_nb_decimals_places(x, max_precision=6):
    """
    Returns the number of decimal places of the scalar `x`
    """
    # ---------------------------------------------------------------------- #
    
    # checking the specified arguments
    
    assert np.isscalar(x)
    
    assert isinstance(max_precision, int)
    assert max_precision >= 1
    
    # ---------------------------------------------------------------------- #
    
    # converting `x` into a positive number, since it doesn't affect its number
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


##############################################################################


# Other functions used to validate values/objects/arguments


def _validate_hash_of_downloaded_raw_MNIST_dataset(path_of_downloaded_data):
    """
    Checks if the hash of the downloaded raw MNIST data is valid or not (mainly
    for security purposes)
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified path
    
    assert isinstance(path_of_downloaded_data, str)
    assert len(path_of_downloaded_data) > 0
    assert len(path_of_downloaded_data.strip()) > 0
    
    assert os.path.exists(path_of_downloaded_data)
    
    # ---------------------------------------------------------------------- #
    
    # default SHA-256 hash value of the downloaded raw MNIST data
    
    default_hash = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
    
    # ---------------------------------------------------------------------- #
    
    # computing the hash value of the specified data
    
    hasher = sha256()
    chunk_size = 65535 # for example
    
    with open(path_of_downloaded_data, "rb") as DOWNLOADED_DATA:
        for chunk in iter(lambda: DOWNLOADED_DATA.read(chunk_size), b""):
            hasher.update(chunk)
    
    computed_hash = hasher.hexdigest()
    
    # ---------------------------------------------------------------------- #
    
    if computed_hash != default_hash:
        raise Exception(f"_validate_hash_of_downloaded_raw_MNIST_dataset (utils.py) - The hash of the raw MNIST data downloaded to the location \"{path_of_downloaded_data}\" is invalid !")


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
    (of the "mnist_dataset.py" script)
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
    
    DEFAULT_NB_CLASSES = 10
    expected_classes = np.arange(DEFAULT_NB_CLASSES)
    assert np.allclose(np.unique(raw_y_train), expected_classes)
    assert np.allclose(np.unique(raw_y_test),  expected_classes)


def _validate_label_vector(y, max_class=9, is_whole_label_vector=True):
    """
    Checks the validity of the label vector `y`
    
    Here, the default value of `max_class` (= 9) is specific to the MNIST dataset
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
        - The string "all" (if you're working with all the digits ranging
          from 0 to 9)
        - A list/tuple/1D-array containing the specific digits you're
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


def _validate_activation_input(x):
    """
    Checks the validity of `x` (as an input of an activation function)
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    """
    assert isinstance(x, np.ndarray)
    assert len(x.shape) in [1, 2]
    
    global DEFAULT_DATATYPE
    check_dtype(x, DEFAULT_DATATYPE)


def _validate_loss_inputs(y_true, y_pred):
    """
    Checks if `y_true` and `y_pred` are valid or not (as inputs of a
    loss function)
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    assert len(y_true.shape) in [1, 2]
    assert y_true.shape == y_pred.shape
    
    global DEFAULT_DATATYPE
    check_dtype(y_true, DEFAULT_DATATYPE)
    check_dtype(y_pred, DEFAULT_DATATYPE)


def _validate_leaky_ReLU_coeff(leaky_ReLU_coeff):
    """
    Checks if `leaky_ReLU_coeff` is valid or not, and returns it (cast
    to `DEFAULT_DATATYPE`)
    """
    assert isinstance(leaky_ReLU_coeff, float)
    
    global DEFAULT_DATATYPE, DTYPE_RESOLUTION
    
    leaky_ReLU_coeff = cast(leaky_ReLU_coeff, DEFAULT_DATATYPE)
    leaky_ReLU_coeff = max(leaky_ReLU_coeff, DTYPE_RESOLUTION)
    
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
    Prints the progress bar related to the download of the raw MNIST data
    
    The signature of this function is imposed by the `reporthook` kwarg of
    the `urllib.request.urlretrieve` method
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
            t_beginning_downloading = perf_counter()
            
            # actually downloading the raw MNIST data from `data_URL`, and
            # saving it to the location `default_path_of_downloaded_data`
            urlretrieve(
                url=data_URL,
                filename=default_path_of_downloaded_data,
                reporthook=_download_progress_bar
            )
            
            assert os.path.exists(default_path_of_downloaded_data)
            
            t_end_downloading = perf_counter()
            duration_downloading = t_end_downloading - t_beginning_downloading
            print(f"\n\nSuccessfully downloaded the raw MNIST data to the location \"{default_path_of_downloaded_data}\". Done in {duration_downloading:.3f} seconds")
        except (Exception, KeyboardInterrupt):
            print("\n")
            if os.path.exists(default_path_of_downloaded_data):
                os.remove(default_path_of_downloaded_data)
            raise
    
    # checking if the hash value of the downloaded data is valid or not (for
    # security purposes)
    _validate_hash_of_downloaded_raw_MNIST_dataset(default_path_of_downloaded_data)
    
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
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    assert isinstance(array, np.ndarray)
    
    assert isinstance(precision, int)
    assert precision >= 0
    
    # ---------------------------------------------------------------------- #
    
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
    # checking the validity of the specified arguments
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


# Functions related to the display of the *styled* confusion matrix


def highlight_diagonal(conf_matrix_as_dataframe, color_of_diagonal="green"):
    """
    Function that is only used by the `print_confusion_matrix` function
    (of this script) if the latter is being run from a Jupyter notebook, and
    if its returned confusion matrix is NOT NORMALIZED. Returns a dataframe
    mask with the CSS property "background-color" in all of its diagonal
    elements, and an empty string everywhere else
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    # checking `conf_matrix_as_dataframe`
    assert isinstance(conf_matrix_as_dataframe, DataFrame)
    conf_matrix_shape = conf_matrix_as_dataframe.shape
    assert len(conf_matrix_shape) == 2
    nb_classes = conf_matrix_shape[0]
    assert nb_classes >= 2
    assert conf_matrix_shape[1] == nb_classes
    conf_matrix_dtype = conf_matrix_as_dataframe.to_numpy().dtype.type
    assert issubclass(conf_matrix_dtype, np.integer)
    
    # checking `color_of_diagonal`
    assert isinstance(color_of_diagonal, str)
    assert len(color_of_diagonal) > 0
    assert " " not in color_of_diagonal
    color_of_diagonal = color_of_diagonal.lower()
    
    # ---------------------------------------------------------------------- #
    
    diagonal_mask = np.full(conf_matrix_shape, "", dtype="<U24")
    
    CSS_property = f"background-color: {color_of_diagonal};"
    np.fill_diagonal(diagonal_mask, CSS_property)
    
    diagonal_mask_as_dataframe = DataFrame(
        diagonal_mask,
        index=conf_matrix_as_dataframe.index,
        columns=conf_matrix_as_dataframe.columns
    )
    
    return diagonal_mask_as_dataframe


def highlight_all_cells(value, colormap=cm.Greens):
    """
    Function that is only used by the `print_confusion_matrix` function
    (of this script) if the latter is being run from a Jupyter notebook, and
    if its returned confusion matrix is NORMALIZED. Returns a CSS property
    "background-color", where the intensity of the associated color (relative
    to the specified colormap) is proportional to `value`
    
    NB : The input `value` is meant to be a percentage (or its string representation)
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    assert isinstance(value, (str, np.str_, float, np.float_))
    if isinstance(value, (str, np.str_)):
        # getting rid of the trailing " %"
        value = float(value[ : -2])
    assert (value >= 0) and (value <= 100) # `value` is a percentage
    
    assert issubclass(type(colormap), Colormap)
    
    # ---------------------------------------------------------------------- #
    
    # this scaling factor scales down the (maximum) intensity of the color,
    # since the most intense colors of the used colormaps are quite dark
    scaling_factor = 0.75
    assert (scaling_factor > 0) and (scaling_factor <= 1)
    
    relative_color_intensity = scaling_factor * (value / 100)
    assert (relative_color_intensity >= 0) and (relative_color_intensity <= scaling_factor)
    
    hex_color_of_cell = rgb2hex(colormap(relative_color_intensity))
    CSS_property = f"background-color: {hex_color_of_cell};"
    
    return CSS_property


def print_confusion_matrix(
        conf_matrix,
        selected_classes="all",
        normalize="no",
        precision=1,
        color="green",
        offset_spacing=1,
        display_with_line_breaks=True
    ):
    """
    Prints the styled confusion matrix. The result won't be the same depending
    on whether you're running the program on a Jupyter notebook or not !
    
    Here, `conf_matrix` is a non-normalized confusion matrix (i.e. a raw,
    integer-valued confusion matrix). By design, `conf_matrix` is meant to
    be the output of the `confusion_matrix` function (of the "core.py" script)
    
    The kwarg `selected_classes` can either be :
        - the string "all" (if you're working with all the digits ranging
          from 0 to 9)
        - a list/tuple/1D-array containing the specific digits you're
          working with (e.g. [2, 4, 7])
    """
    # ====================================================================== #
    
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
    possible_values_for_normalize_kwarg = ["no", "columns", "rows"]
    if normalize not in possible_values_for_normalize_kwarg:
        raise ValueError(f"get_confusion_matrix_as_dataframe (utils.py) - Unrecognized value for the `normalize` kwarg : \"{normalize}\" (possible values : {list_to_string(possible_values_for_normalize_kwarg)})")
    
    if normalize != "no":
        assert isinstance(precision, int)
        assert precision >= 0
    
    assert isinstance(color, str)
    assert len(color) > 0
    assert " " not in color
    color = color.lower()
    
    # keys   : generic color names
    # values : (
    #     associated ANSI escape sequences (for background colors),
    #     associated CSS/HTML color names,
    #     associated Matplotlib colormaps
    # )
    COLORS_AND_COLORMAPS = {
        "green"  : (Back.GREEN,   "green",     cm.Greens),
        "blue"   : (Back.BLUE,    "cyan",      cm.Blues),   # `Back.BLUE` and `Back.CYAN` both work here
        "purple" : (Back.MAGENTA, "indigo",    cm.Purples),
        "red"    : (Back.RED,     "red",       cm.Reds),
        "orange" : (Back.YELLOW,  "orangered", cm.Oranges)
    }
    
    if color not in COLORS_AND_COLORMAPS:
        raise ValueError(f"print_confusion_matrix (utils.py) - Unrecognized value for the `color` kwarg : \"{color}\" (possible color names : {list_to_string(list(COLORS_AND_COLORMAPS.keys()))})")
    
    jupyter_notebook = is_being_run_on_jupyter_notebook()
    
    if not(jupyter_notebook):
        assert isinstance(offset_spacing, int)
        assert offset_spacing >= 0
        
        assert isinstance(display_with_line_breaks, bool)
        set_option("display.expand_frame_repr", display_with_line_breaks) # option of the Pandas module
    
    # ====================================================================== #
    
    # setting some display options of the Pandas module (for convenience purposes)
    
    set_option("display.max_rows",     None)
    set_option("display.max_columns",  None)
    set_option("display.width",        None)
    set_option("display.max_colwidth", None)
    
    # ====================================================================== #
    
    # converting the confusion matrix into its associated dataframe
    
    if normalize != "no":
        normalized_conf_matrix = np.float_(np.copy(conf_matrix))
        
        for class_index in range(nb_classes):
            if normalize == "columns":
                # computing the PRECISION of the (predicted) class at column index `class_index`
                sum_of_column = np.sum(normalized_conf_matrix[:, class_index])
                if np.allclose(sum_of_column, 0.0):
                    normalized_conf_matrix[:, class_index] = 0
                else:
                    normalized_conf_matrix[:, class_index] /= sum_of_column
            
            elif normalize == "rows":
                # computing the RECALL of the (true) class at row index `class_index`
                sum_of_row = np.sum(normalized_conf_matrix[class_index, :])
                if np.allclose(sum_of_row, 0.0):
                    raise Exception(f"print_confusion_matrix (utils.py) - The true class \"{class_names[class_index]}\" (class_index={class_index}) isn't represented in the confusion matrix !")
                normalized_conf_matrix[class_index, :] /= sum_of_row
        
        normalized_conf_matrix = np.round(100 * normalized_conf_matrix, precision)
        
        conf_matrix_as_dataframe = DataFrame(
            normalized_conf_matrix,
            index=class_names,
            columns=class_names
        )
        
        for class_index, class_name in enumerate(class_names):
            # adding the "%" symbol to all the elements of the dataframe
            conf_matrix_as_dataframe[class_name] = conf_matrix_as_dataframe[class_name].astype("<U24") + " %"
    else:
        # here, the `normalize` kwarg is equal to "no", therefore the raw
        # confusion matrix will be printed
        conf_matrix_as_dataframe = DataFrame(
            conf_matrix,
            index=class_names,
            columns=class_names
        )
    
    # by definition
    conf_matrix_as_dataframe.columns.name = "PREDICTED"
    conf_matrix_as_dataframe.index.name   = "ACTUAL"
    
    # printing the header/description of the confusion matrix
    if normalize == "no":
        conf_matrix_header = "\nRAW CONFUSION MATRIX"
    elif normalize == "columns":
        conf_matrix_header = "\nNETWORK PRECISION - NORMALIZED CONFUSION MATRIX"
    elif normalize == "rows":
        conf_matrix_header = "\nNETWORK RECALL - NORMALIZED CONFUSION MATRIX"
    conf_matrix_header += f" (normalized=\"{normalize}\") :"
    print(conf_matrix_header)
    
    # ====================================================================== #
    
    # styling the confusion matrix (on Jupyter notebook only)
    
    if jupyter_notebook:
        color_of_diagonal = COLORS_AND_COLORMAPS[color][1]
        colormap          = COLORS_AND_COLORMAPS[color][2]
        
        if normalize == "no":
            # if the returned confusion matrix isn't normalized, then
            # we'll only highlight its diagonal cells
            conf_matrix_styler = conf_matrix_as_dataframe.style.apply(
                highlight_diagonal,
                axis=None,
                color_of_diagonal=color_of_diagonal
            )
        else:
            # if the returned confusion matrix is normalized, then its cells
            # will be highlighted in a color whose intensity is proportional
            # to the value they hold (relative to the used colormap)
            conf_matrix_styler = conf_matrix_as_dataframe.style.applymap(
                highlight_all_cells,
                colormap=colormap
            )
        
        # ------------------------------------------------------------------ #
        
        # defining the color that will highlight the hovered cells of the
        # styled confusion matrix
        
        relative_color_intensity = 0.50
        assert (relative_color_intensity > 0) and (relative_color_intensity <= 1)
        
        hex_color_of_hovered_cells = rgb2hex(colormap(relative_color_intensity))
        
        CSS_property_cell_hovering = {
            "selector" : "td:hover",
            "props"    : [("background-color", hex_color_of_hovered_cells)]
        }
        
        conf_matrix_styler.set_table_styles([CSS_property_cell_hovering])
        
        # ------------------------------------------------------------------ #
        
        # NB : `conf_matrix_styler` is a Pandas Styler object (i.e. a
        #       `pandas.io.formats.style.Styler` object), NOT a Pandas
        #       DataFrame object
        display(conf_matrix_styler)
        
        return
    
    # ====================================================================== #
    
    # converting the confusion matrix (as a Pandas DataFrame) to a string,
    # and adding the offset spacing (if not on Jupyter notebook)
    
    offset_spacing = " " * offset_spacing
    str_conf_matrix_as_dataframe = f"\n{offset_spacing}" + str(conf_matrix_as_dataframe).replace("\n", f"\n{offset_spacing}") + "\n"
    
    # ====================================================================== #
    
    # Highlighting the diagonal values with a specific color (if not on
    # Jupyter notebook). Unfortunately, this has to be done AFTER the confusion
    # matrix (as a Pandas DataFrame) is converted to a string, so that the
    # Pandas module can keep the columns of the DataFrame aligned. Indeed, the
    # string representations (i.e. the `repr` values) of the ANSI escape codes
    # of the used colors are (quite naturally) not empty strings ! Here, the
    # text parsing is quite tedious, but it's necessary
    
    # the diagonal of the confusion matrix will be printed in this color
    printed_color_of_diagonal = COLORS_AND_COLORMAPS[color][0]
    
    escape_sequence_of_color_reset = Style.RESET_ALL
    
    lines_of_str_conf_matrix_as_dataframe = str_conf_matrix_as_dataframe.split("\n")
    lines_of_str_colored_conf_matrix_as_dataframe = lines_of_str_conf_matrix_as_dataframe.copy()
    
    max_len_of_class_names = max([len(class_name) for class_name in class_names])
    
    # first index (inclusive) of the area of interest of each row (i.e. the
    # area of each row that potentially contains the actual numerical data of
    # the confusion matrix)
    first_index_area_of_interest = len(offset_spacing) + max(len(conf_matrix_as_dataframe.columns.name), len(conf_matrix_as_dataframe.index.name), max_len_of_class_names) + 2
    
    for row_index, current_row in enumerate(lines_of_str_conf_matrix_as_dataframe):
        beginning_of_row = current_row[ : first_index_area_of_interest]
        
        # ------------------------------------------------------------------ #
        
        # checking if the current row is a row of interest or not (if not, the
        # current row is skipped)
        
        content_of_beginning_of_row = beginning_of_row.split()
        if len(content_of_beginning_of_row) == 0:
            continue
        
        assert len(content_of_beginning_of_row) == 1
        content_of_beginning_of_row = content_of_beginning_of_row[0]
        
        if content_of_beginning_of_row == conf_matrix_as_dataframe.index.name:
            continue
        
        if content_of_beginning_of_row == conf_matrix_as_dataframe.columns.name:
            # updating the current column indices (for classes)
            
            current_class_indices_in_columns = current_row.split()[1 : ]
            if current_class_indices_in_columns[-1] == "\\":
                # here, the trailing "\" means that there's a line break
                current_class_indices_in_columns = current_class_indices_in_columns[ : -1]
            
            # converting the column indices into integers
            current_class_indices_in_columns = [int(str_class_index_in_columns) for str_class_index_in_columns in current_class_indices_in_columns]
            
            continue
        
        # updating the current row index (for classes)
        assert content_of_beginning_of_row in class_names
        current_class_index_in_rows = int(content_of_beginning_of_row)
        
        # ------------------------------------------------------------------ #
        
        # if we made it to this point, then it means that the current row
        # is a row of interest (i.e. it holds the actual numerical data of
        # the confusion matrix)
        
        area_of_interest = current_row[first_index_area_of_interest : ]
        
        split_area_of_interest = area_of_interest.split(" ")
        split_colored_area_of_interest = split_area_of_interest.copy()
        
        # ------------------------------------------------------------------ #
        
        # this counter holds the number of numerical values that have been
        # encountered in the area of interest of the current row
        nb_seen_numerical_values_in_row = 0
        
        for element_index, element in enumerate(split_area_of_interest):
            # here, `element_is_numerical_value` is a boolean indicating whether
            # `element` is an integer or a float (i.e. a numerical value)
            try:
                _ = float(element)
                element_is_numerical_value = True
            except:
                element_is_numerical_value = False
            
            if element_is_numerical_value:
                nb_seen_numerical_values_in_row += 1
                
                # updating the current column index (for classes)
                current_class_index_in_columns = current_class_indices_in_columns[nb_seen_numerical_values_in_row - 1]
            
            if (current_class_index_in_columns == current_class_index_in_rows) and (element_is_numerical_value or (element == "%")):
                # here, `element` is on the diagonal of the confusion matrix,
                # therefore color will be added to it !
                
                if element_is_numerical_value:
                    try:
                        next_element_is_a_percent_symbol = (split_colored_area_of_interest[element_index + 1] == "%")
                    except:
                        next_element_is_a_percent_symbol = False
                    
                    colored_element = printed_color_of_diagonal + element
                    if not(next_element_is_a_percent_symbol):
                        colored_element += escape_sequence_of_color_reset
                    split_colored_area_of_interest[element_index] = colored_element
                    
                    if not(next_element_is_a_percent_symbol):
                        break
                else:
                    assert element == "%"
                    split_colored_area_of_interest[element_index] += escape_sequence_of_color_reset
                    
                    break
        
        # replacing the area of interest with its colored counterpart
        colored_area_of_interest = " ".join(split_colored_area_of_interest)
        
        # ------------------------------------------------------------------ #
        
        lines_of_str_colored_conf_matrix_as_dataframe[row_index] = beginning_of_row + colored_area_of_interest
    
    # getting the string representation of the confusion matrix (as a Pandas
    # DataFrame) with a colored diagonal 
    str_colored_conf_matrix_as_dataframe = "\n".join(lines_of_str_colored_conf_matrix_as_dataframe)
    
    # ====================================================================== #
    
    # actually printing the confusion matrix with a colored diagonal
    print(str_colored_conf_matrix_as_dataframe)

