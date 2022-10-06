# -*- coding: utf-8 -*-

"""
Script defining some miscellaneous useful functions
"""

from __future__ import annotations

import os
from time import perf_counter, time
from urllib.request import urlretrieve
from hashlib import sha256
from typing import Union, Optional

try:
    from IPython.display import display
except ImportError:
    # The `IPython.display.display` method will only be called if the
    # `print_confusion_matrix` function (of this script) is run from a
    # Jupyter notebook, which itself requires an IPython backend. Therefore,
    # if the import fails, then that automatically implies that the IPython
    # backend is not available, which in turn implies that the code is not
    # being run on a Jupyter notebook, meaning that the `IPython.display.display`
    # method would have never been called anyway !
    display = None

import numpy as np
from pandas import DataFrame, set_option

# imports related to Matplotlib colormaps
from matplotlib import cm
from matplotlib.colors import Colormap, rgb2hex

# used to print colored text in a Python console/terminal
from colorama import init, Back, Style


##############################################################################


# Defining the datatype of ALL the data that will flow through the network
# (i.e. we're defining the "global datatype"), along with its resolution

DEFAULT_DATATYPE = "float32"                             # = "float32" or "float64"
DTYPE_RESOLUTION = np.finfo(DEFAULT_DATATYPE).resolution # = 1e-06 or 1e-15 (respectively)


##############################################################################


# Function related to pure datatype checking (for numeric NumPy arrays)


def _validate_numpy_dtype(dtype: Union[str, type, np.dtype]) -> np.dtype:
    """
    Checks if the specified datatype is a valid (numeric) NumPy datatype or not
    """
    if isinstance(dtype, str):
        dtype = dtype.lower().replace(" ", "")
    
    try:
        numpy_dtype = np.dtype(dtype)
    except:
        if isinstance(dtype, type):
            str_datatype = dtype.__name__
        else:
            str_datatype = str(dtype)
        raise ValueError(f"_validate_numpy_datatype (utils.py) - Invalid NumPy datatype : \"{str_datatype}\"")
    
    if not(issubclass(numpy_dtype.type, np.number)):
        raise TypeError(f"_validate_numpy_datatype (utils.py) - The specified NumPy datatype (= \"{str(numpy_dtype)}\") isn't numeric !")
    
    return numpy_dtype


##############################################################################


# Functions related to the checking and setting of the global datatype


def _validate_global_datatype(datatype: Union[str, type, np.dtype]) -> np.dtype:
    """
    Checks if the specified datatype can be the "global datatype" of the
    network (or not). For now, the only available datatypes are float32
    and float64
    """
    datatype = _validate_numpy_dtype(datatype)
    
    AVAILABLE_DATATYPES = (
        np.float32,
        np.float64
    )
    
    if datatype.type not in AVAILABLE_DATATYPES:
        raise ValueError(f"_validate_global_datatype (utils.py) - Unrecognized value for `datatype` : \"{str(datatype)}\" (has to be float32 or float64)")
    
    return datatype


def set_global_datatype(datatype: Union[str, type, np.dtype]) -> None:
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
    # checking the specified datatype first
    datatype = _validate_global_datatype(datatype)
    
    global DEFAULT_DATATYPE, DTYPE_RESOLUTION
    DEFAULT_DATATYPE = datatype
    DTYPE_RESOLUTION = np.finfo(DEFAULT_DATATYPE).resolution


##############################################################################


# Functions related to the checking/casting of the datatypes of (numeric)
# NumPy arrays and scalars


def check_dtype(
        x : Union[int, float, np.ndarray],
        dtype: Union[str, type, np.dtype]
    ) -> None:
    """
    Checks if the NumPy datatype of `x` is `dtype` or not. Here, `x` can either
    be a scalar or a vector/matrix. By design, in most cases, `dtype` will be
    equal to `DEFAULT_DATATYPE`
    
    NB : This function will NOT work if you're trying to check if a scalar is
         an instance of a NATIVE type (like `int` or `float` for instance)
    """
    # checking the specified arguments
    assert np.isscalar(x) or isinstance(x, np.ndarray)
    dtype = _validate_numpy_dtype(dtype)
    
    if np.isscalar(x):
        if type(x) != dtype.type:
            raise TypeError(f"check_dtype (utils.py) - The datatype of the scalar `x` isn't \"{str(dtype)}\", it's \"{type(x).__name__}\" !")
    else:
        # here, `x` is a vector/matrix
        if x.dtype != dtype:
            raise TypeError(f"check_dtype (utils.py) - The datatype of the array `x` isn't \"{str(dtype)}\", it's \"{str(x.dtype)}\" !")


def cast(
        x : Union[int, float, np.ndarray],
        dtype: Union[str, type, np.dtype]
    ) -> Union[int, float, np.ndarray]:
    """
    Returns the cast version of `x` to `dtype`. Here, `x` can either be a
    scalar or a vector/matrix. By design, in most cases, `dtype` will be
    equal to `DEFAULT_DATATYPE`
    """
    # checking the specified arguments
    assert np.isscalar(x) or isinstance(x, np.ndarray)
    dtype = _validate_numpy_dtype(dtype)
    
    if np.isscalar(x):
        cast_x = dtype.type(x)
    else:
        # here, `x` is a vector/matrix
        cast_x = x.astype(dtype)
    
    return cast_x


##############################################################################


# Generally useful functions


def vector_to_categorical(
        y: np.ndarray,
        *,
        dtype: Union[str, type, np.dtype] = int
    ) -> np.ndarray:
    """
    Performs one-hot encoding on a 1D vector of INTEGER labels
    """
    # checking the validity of the specified arguments
    _validate_label_vector(y)
    dtype = _validate_numpy_dtype(dtype)
    
    nb_labels = y.size
    
    distinct_labels, distinct_labels_inverse = np.unique(y, return_inverse=True)
    nb_classes = distinct_labels.size
    assert nb_classes >= 2
    
    y_categorical = np.zeros((nb_labels, nb_classes), dtype=dtype)
    
    for label_index, label in enumerate(distinct_labels_inverse):
        # by definition
        y_categorical[label_index, label] = 1
    
    return y_categorical


def categorical_to_vector(
        y_categorical: np.ndarray,
        *,
        enable_checks: bool = True
    ) -> np.ndarray:
    """
    Converts a 2D categorical matrix (one-hot encoded matrix or logits) into
    its associated 1D vector of INTEGER labels
    """
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        # checking the validity of `y_categorical`
        assert isinstance(y_categorical, np.ndarray)
        assert y_categorical.ndim == 2
        global DEFAULT_DATATYPE
        check_dtype(y_categorical, DEFAULT_DATATYPE)
    
    # by definition
    y = np.argmax(y_categorical, axis=1)
    
    if enable_checks:
        _validate_label_vector(y, is_whole_label_vector=False)
    return y


def list_to_string(L: list) -> str:
    """
    Converts the specified non-empty list (or tuple or 1D numpy array) into
    a string
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the input `L`
    
    if not(isinstance(L, (list, tuple, np.ndarray))):
        raise TypeError(f"list_to_string (utils.py) - The input `L` isn't a list (or a tuple or a 1D numpy array), it's a \"{L.__class__.__name__}\" !")
    
    if isinstance(L, np.ndarray):
        assert L.ndim == 1, "list_to_string (utils.py) - The input `L` has to be 1-dimensional !"
    
    nb_elements = len(L)
    assert nb_elements > 0, "list_to_string (utils.py) - The input `L` is empty !"
    
    # ---------------------------------------------------------------------- #
    
    # initializing the string representation of the specified list
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


ID = int(time()) # global variable, will only be initialized *once*
def generate_unique_ID() -> int:
    """
    Generates a unique identifier (as an integer)
    """
    global ID
    ID += 1
    return ID


def clear_currently_printed_row(*, max_size_of_row: int = 150) -> None:
    """
    Clears the currently printed row, and sets the pointer of the `print`
    function at the very beginning of that same row
    """
    # checking the specified argument
    assert isinstance(max_size_of_row, int)
    assert max_size_of_row >= 0
    
    blank_row = " " * max_size_of_row
    print(blank_row, end="\r")


def progress_bar(
        current_index: int,
        total_nb_elements: int,
        *,
        progress_bar_size: int = 15,
        enable_checks: bool = True
    ) -> str:
    """
    Returns a progress bar (as a string) corresponding to the progress of
    `current_index` relative to `total_nb_elements`. The value of the kwarg
    `progress_bar_size` indicates how long the content of the progress bar is
    
    For example, `progress_bar(70, 100, progress_bar_size=10)` will return
    the following string : "[======>...]"
    """
    # ---------------------------------------------------------------------- #
    
    # checking the specified arguments
    
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
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
    
    if enable_checks:
        assert len(str_progress_bar) == progress_bar_size + 2
    
    return str_progress_bar


def standardize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalizes a matrix such that its mean is 0 and its standard deviation
    is 1 (i.e. it returns the standardized version of the matrix). By default,
    if `data` is a 2D matrix, then it will be standardized along its ROWS
    
    Here, `data` can either be a 1D vector or a 2D matrix
    """
    # checking the validity of `data`
    assert isinstance(data, np.ndarray)
    assert data.ndim in [1, 2]
    assert data.shape[0] > 0
    global DEFAULT_DATATYPE
    check_dtype(data, DEFAULT_DATATYPE)
    
    # checking if there are any samples that have a standard deviation of zero
    # (in that case, it means that the illegal sample is filled with the exact
    # same constant value)
    data_std = data.std(axis=-1, keepdims=True)
    for sample_std in data_std.flatten():
        assert not(np.allclose(sample_std, 0))
    
    # actually standardizing the data
    standardized_data = (data - data.mean(axis=-1, keepdims=True)) / data_std
    
    assert standardized_data.shape == data.shape
    check_dtype(standardized_data, DEFAULT_DATATYPE)
    
    return standardized_data


def basic_split(
        sample_indices: Union[np.ndarray, list, tuple],
        nb_train_samples: int,
        nb_test_samples:  int,
        *,
        seed: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Does a basic split of the specified 1D array (or list/tuple) of sample
    indices into 2 randomly generated train and test subsets
    """
    # ---------------------------------------------------------------------- #
    
    # Checking the specified arguments
    
    assert isinstance(sample_indices, (np.ndarray, list, tuple))
    if isinstance(sample_indices, (list, tuple)):
        sample_indices = np.array(sample_indices)
    assert sample_indices.ndim == 1
    nb_samples = sample_indices.size
    assert issubclass(sample_indices.dtype.type, np.integer)
    assert np.min(sample_indices) >= 0
    distinct_sample_indices = np.unique(sample_indices)
    assert distinct_sample_indices.size == nb_samples
    
    assert isinstance(nb_train_samples, int)
    assert nb_train_samples > 0
    
    assert isinstance(nb_test_samples, int)
    assert nb_test_samples > 0
    
    assert nb_train_samples + nb_test_samples <= nb_samples
    
    assert isinstance(seed, (type(None), int))
    if seed is not None:
        assert seed >= 0
    
    # ---------------------------------------------------------------------- #
    
    np.random.seed(seed)
    
    train_indices = np.random.choice(
        sample_indices,
        size=(nb_train_samples, ),
        replace=False
    )
    
    potential_test_indices = []
    for sample_index in sample_indices:
        if sample_index not in train_indices:
            potential_test_indices.append(sample_index)
    
    test_indices = np.random.choice(
        potential_test_indices,
        size=(nb_test_samples, ),
        replace=False
    )
    
    np.random.seed(None) # resetting the seed
    
    # ---------------------------------------------------------------------- #
    
    return train_indices, test_indices


def is_being_run_on_jupyter_notebook() -> bool:
    """
    Returns a boolean indicating whether the code is currently being run on
    a Jupyter notebook or not
    """
    try:
        current_IPython_shell_name = get_ipython().__class__.__name__
        jupyter_notebook = (current_IPython_shell_name == "ZMQInteractiveShell")
        
        global display
        if jupyter_notebook and (display is None):
            # cf. the note at the very beginning of this script
            raise ImportError("The `display` method (from the `IPython.display` module) should have been imported, since, in theory, the code is being run on a Jupyter notebook (which itself runs on an IPython backend) !")
        
        return jupyter_notebook
    except (Exception, UnboundLocalError, NameError):
        return False


def count_nb_decimals_places(
        x: Union[int, float],
        *,
        max_precision: int = 9
    ) -> int:
    """
    Returns the number of decimal places of the real number `x`
    
    If the fractional part of `x` is non-zero and is either :
        - stricly less than `10**(-max_precision) / 2`
        - strictly higher than `1 - 10**(-max_precision) / 2`
    then `0` will be returned
    If `x` is an integer, then `0` will also be returned
    """
    # ---------------------------------------------------------------------- #
    
    # checking the specified arguments
    
    assert np.isscalar(x)
    assert np.isreal(x)
    
    assert isinstance(max_precision, int)
    assert max_precision >= 1
    
    # ---------------------------------------------------------------------- #
    
    # converting `x` into a positive number, since it doesn't affect its number
    # of decimal places
    x = float(abs(x))
    
    # converting `x` into its fractional part, since it also doesn't affect
    # its number of decimal places (by definition)
    x -= int(x)
    assert (x >= 0) and (x < 1)
    
    # rounding (the fractional part of) `x` to the specified precision
    x = round(x, max_precision)
    
    if x == 0:
        # in this case, the original `x` was either an integer, or had a (non-zero)
        # fractional part that was either strictly less than `10**(-max_precision) / 2`
        # or strictly higher than `1 - 10**(-max_precision) / 2`
        nb_decimal_places = 0
    else:
        # here, `x` is a float in the range ]0, 1[, or, more precisely, in the
        # range [10**(-max_precision) / 2, 1 - 10**(-max_precision) / 2]
        assert (x >= 10**(-max_precision) / 2) and (x <= 1 - 10**(-max_precision) / 2)
        
        # we're doing this just in case the string representation of `x` uses
        # the scientific notation (like `1e-05` for instance)
        str_x = f"{x:.{max_precision}f}".rstrip("0")
        assert len(str_x) >= 3
        
        # getting rid of (the information related to the) the leading "0.",
        # which has a length of 2
        nb_decimal_places = len(str_x) - 2
    
    return nb_decimal_places


def get_dtype_of_array(array: np.ndarray) -> str:
    """
    Returns the datatype of an array (as a string)
    """
    assert isinstance(array, np.ndarray)
    str_dtype = "numpy." + str(array.dtype)
    return str_dtype


def get_range_of_array(
        array: np.ndarray,
        *,
        precision: int = 3
    ) -> str:
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
    
    array_dtype = array.dtype.type
    
    if issubclass(array_dtype, np.floating):
        # floating-point data
        str_range = f"{min_element_of_array:.{precision}f} -> {max_element_of_array:.{precision}f}"
    elif issubclass(array_dtype, np.integer):
        # integer data
        str_range = f"{min_element_of_array} -> {max_element_of_array}"
    else:
        raise TypeError(f"get_range_of_array (utils.py) - The input `array` has an unrecognized datatype : \"{array_dtype.__name__}\" (it has to be numeric, i.e. float or int)")
    
    return str_range


def display_class_distributions(
        dict_of_label_vectors: dict[str, np.ndarray],
        *,
        selected_classes: Union[str, list[int]] = "all",
        dict_of_real_class_names: Optional[dict[int, str]] = None,
        precision: int = 2
    ) -> None:
    """
    Prints the class distributions of the specified label vectors (i.e. the
    values of the dictionary `dict_of_label_vectors`). The keys of `dict_of_label_vectors`
    are the names of the corresponding label vectors (as strings)
    
    The kwarg `selected_classes` can either be :
        - the string "all", if you're working with all the classes (default)
        - a list/tuple/1D-array containing the specific class indices you're
          working with (e.g. [2, 4, 7])
    
    The kwarg `dict_of_real_class_names`, if not set to `None`, is a dictionary
    with :
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
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    assert isinstance(dict_of_label_vectors, dict)
    assert len(dict_of_label_vectors) > 0
    nb_classes = None
    for label_vector_name, label_vector in dict_of_label_vectors.items():
        assert isinstance(label_vector_name, str)
        assert len(label_vector_name.strip()) > 0
        label_vector_name = label_vector_name.strip()
        
        _validate_label_vector(label_vector)
        if nb_classes is None:
            distinct_class_indices = np.unique(label_vector)
            nb_classes = distinct_class_indices.size
        else:
            assert np.allclose(np.unique(label_vector), distinct_class_indices)
    
    selected_classes, class_names = _validate_selected_classes(
        selected_classes,
        nb_classes,
        dict_of_real_class_names=dict_of_real_class_names
    )
    
    assert isinstance(precision, int)
    assert precision >= 0
    
    # ---------------------------------------------------------------------- #
    
    if dict_of_real_class_names is not None:
        displayed_string = "\nClass distributions :\n"
    else:
        displayed_string = "\nDistribution of the class indices :\n"
    
    # used to align the displayed class names
    max_length_of_class_names = max([len(class_name) for class_name in class_names])
    
    for label_vector_name, label_vector in dict_of_label_vectors.items():
        nb_labels = label_vector.size
        
        displayed_string += f"\n{label_vector_name} :"
        
        for class_index, class_name in enumerate(class_names):
            nb_corresponding_class_indices = np.where(label_vector == class_index)[0].size
            proportion = 100 * float(nb_corresponding_class_indices) / nb_labels
            str_proportion = f"{proportion:.{precision}f}"
            if proportion < 10:
                str_proportion = "0" + str_proportion
            
            class_name_spacing = " " * (max_length_of_class_names - len(class_name))
            displayed_string += f"\n    {class_name_spacing}{class_name} --> {str_proportion} %"
    
    print(displayed_string)


##############################################################################


# Input validators


def _validate_label_vector(
        y: np.ndarray,
        *,
        is_whole_label_vector: bool = True
    ) -> None:
    """
    Checks the validity of the label vector `y`
    
    Here, the default value of `max_class` (= 9) is specific to the MNIST dataset
    """
    assert isinstance(is_whole_label_vector, bool)
    
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    assert y.size > 0
    assert issubclass(y.dtype.type, np.integer)
    
    assert np.min(y) >= 0
    
    if is_whole_label_vector:
        nb_classes = np.unique(y).size
        assert nb_classes >= 2


def _validate_selected_classes(
        selected_classes: Union[str, list[int]],
        max_nb_classes: int,
        *,
        dict_of_real_class_names: Optional[dict[int, str]] = None,
    ) -> tuple[Union[str, np.ndarray], list[str]]:
    """
    Checks the validity of the `selected_classes` argument, and returns the
    potentially corrected version of `selected_classes`
    
    Here, `selected_classes` can either be :
        - The string "all", if you're working with all the classes (default)
        - A list/tuple/1D-array containing the specific class indices you're
          working with (e.g. [2, 4, 7])
    
    The kwarg `dict_of_real_class_names`, if not set to `None`, is a dictionary
    with :
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
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of `max_nb_classes`
    
    assert isinstance(max_nb_classes, int)
    assert max_nb_classes >= 2
    
    # ---------------------------------------------------------------------- #
    
    # checking the validity of `selected_classes`
    
    # by default
    class_indices = np.arange(max_nb_classes)
    
    if isinstance(selected_classes, str):
        selected_classes = selected_classes.strip().lower()
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
        assert distinct_selected_classes.size <= max_nb_classes
        
        selected_classes = distinct_selected_classes
        class_indices = selected_classes
    
    # by default
    class_names = [str(class_index) for class_index in class_indices]
    
    # ---------------------------------------------------------------------- #
    
    # checking the validity of `dict_of_real_class_names`
    
    assert isinstance(dict_of_real_class_names, (type(None), dict))
    
    if dict_of_real_class_names is not None:
        if len(dict_of_real_class_names) != len(class_names):
            raise ValueError(f"_validate_selected_classes (utils.py) - The length of the specified `dict_of_real_class_names` kwarg (= {len(dict_of_real_class_names)}) has to be the same as the number of (distinct) selected classes (= {len(class_names)}) !")
        
        for class_index, real_class_name in dict_of_real_class_names.items():
            assert isinstance(class_index, int)
            assert class_index >= 0
            assert str(class_index) in class_names
            
            assert isinstance(real_class_name, str)
            assert len(real_class_name.strip()) > 0
            real_class_name = real_class_name.strip()
            real_class_name = " ".join(real_class_name.split())
            dict_of_real_class_names[class_index] = real_class_name
        
        # sorting `dict_of_real_class_names` such that its keys are in
        # ASCENDING order
        sorted_dict_of_real_class_names = {}
        for sorted_class_index in sorted(dict_of_real_class_names):
            sorted_class_name = dict_of_real_class_names[sorted_class_index]
            sorted_dict_of_real_class_names[sorted_class_index] = sorted_class_name
        
        # updating `class_names` with its real labels
        class_names = list(sorted_dict_of_real_class_names.values())
    
    # ---------------------------------------------------------------------- #
    
    return selected_classes, class_names


def _validate_one_hot_encoded_array(array: np.ndarray) -> None:
    """
    Checks if the specified array is one-hot encoded or not
    
    The input `array` can either be a 1D vector or a 2D matrix (usually the latter) 
    """
    assert isinstance(array, np.ndarray)
    assert array.ndim in [1, 2]
    _validate_numpy_dtype(array.dtype) # checking if `array` has numeric data
    nb_classes = array.shape[-1]
    assert nb_classes >= 2
    
    # checking if the array only contains zeros and ones
    assert np.allclose(np.unique(array), [0, 1]), "The specified array doesn't only contain zeros and ones !"
    
    # checking that there is only one `1` per row (assuming the previous
    # condition was met)
    assert np.allclose(np.sum((array.sum(axis=-1) - 1)**2), 0), "The specified array has rows that don't contain exactly one `1` in them !"


def _validate_split_data_into_batches_inputs(
        data: np.ndarray,
        batch_size: int,
        labels: Union[None, np.ndarray],
        nb_shuffles: int,
        seed: Union[None, int]
    ) -> None:
    """
    Checks the validity of the specified arguments (as inputs of the functions
    used to split the data into batches, defined in this script)
    """
    # checking the validity of the argument `data`
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    nb_samples, nb_features_per_sample = data.shape
    assert nb_features_per_sample >= 2
    
    assert isinstance(batch_size, int)
    assert (batch_size >= 1) and (batch_size <= nb_samples)
    
    # checking the validity of the `labels` kwarg
    assert isinstance(labels, (type(None), np.ndarray))
    if labels is not None:
        assert labels.ndim in [1, 2]
        if labels.ndim == 1:
            _validate_label_vector(labels)
        elif labels.ndim == 2:
            global DEFAULT_DATATYPE
            check_dtype(labels, DEFAULT_DATATYPE)
            _validate_one_hot_encoded_array(labels)
        assert labels.shape[0] == nb_samples
    
    assert isinstance(nb_shuffles, int)
    assert nb_shuffles >= 0
    
    if nb_shuffles > 0:
        assert isinstance(seed, (type(None), int))
        if seed is not None:
            assert seed >= 0


def _validate_activation_input(x: np.ndarray) -> None:
    """
    Checks the validity of `x` (as an input of an activation function)
    
    The input `x` can either be a 1D vector or a 2D matrix (usually the latter)
    """
    assert isinstance(x, np.ndarray)
    assert x.ndim in [1, 2]
    
    global DEFAULT_DATATYPE
    check_dtype(x, DEFAULT_DATATYPE)


def _validate_loss_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Checks if `y_true` and `y_pred` are valid or not (as inputs of a
    loss function)
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    assert y_true.ndim in [1, 2]
    assert y_true.shape == y_pred.shape
    
    global DEFAULT_DATATYPE
    check_dtype(y_true, DEFAULT_DATATYPE)
    check_dtype(y_pred, DEFAULT_DATATYPE)


def _validate_leaky_ReLU_coeff(leaky_ReLU_coeff: float) -> float:
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


# Functions related to the downloading of online data


def _validate_hash_of_downloaded_data(
        path_of_downloaded_data: str,
        hash_value: str
    ) -> None:
    """
    Checks if the (SHA-256) hash of the data that was downloaded to the
    specified path is valid or not, i.e. if it's the same as the specified
    (SHA-256) hash value
    
    This function is mainly used for security purposes
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified path
    
    assert isinstance(path_of_downloaded_data, str)
    assert len(path_of_downloaded_data.strip()) > 0
    path_of_downloaded_data = path_of_downloaded_data.strip()
    assert os.path.exists(path_of_downloaded_data)
    
    assert isinstance(hash_value, str)
    assert len(hash_value.strip()) > 0
    hash_value = hash_value.strip()
    assert len(hash_value) == 64 # specific to the SHA-256 hashing algorithm
    
    # ---------------------------------------------------------------------- #
    
    # computing the hash value of the specified data
    
    hasher = sha256()
    chunk_size = 65536 # for example
    
    with open(path_of_downloaded_data, "rb") as DOWNLOADED_DATA:
        while True:
            chunk = DOWNLOADED_DATA.read(chunk_size)
            if chunk == b"":
                break
            hasher.update(chunk)
    
    computed_hash = hasher.hexdigest()
    
    assert isinstance(computed_hash, str)
    assert len(computed_hash) == 64 # specific to the SHA-256 hashing algorithm
    
    # ---------------------------------------------------------------------- #
    
    if computed_hash != hash_value:
        raise Exception(f"_validate_hash_of_downloaded_data (utils.py) - The hash of the data that was downloaded to the location \"{path_of_downloaded_data}\" is invalid !")


def _download_progress_bar(
        block_index: int,
        block_size_in_bytes: int,
        total_size_of_data_in_bytes: int
    ) -> None:
    """
    Prints the progress bar related to the online downloading of data, using
    the `urllib.request.urlretrieve` method
    
    NB : The signature of this function is imposed by the `reporthook` kwarg
         of the `urllib.request.urlretrieve` method
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
    
    # Defining the (default) number of times the progress bar will be updated
    # during the entire download. Don't set this value to a value >= 10,
    # otherwise Python's standard output will bug
    nb_progress_bar_updates = 10
    
    block_index_update_step = total_nb_blocks // nb_progress_bar_updates
    
    if (block_index % block_index_update_step == 0) or (block_index == 1) or (block_index == total_nb_blocks):
        size_of_already_downloaded_data_in_bytes = block_index * block_size_in_bytes
        
        if block_index == total_nb_blocks:
            assert size_of_already_downloaded_data_in_bytes >= total_size_of_data_in_bytes
            size_of_already_downloaded_data_in_bytes = total_size_of_data_in_bytes
        
        current_progress_bar = progress_bar(
            size_of_already_downloaded_data_in_bytes,
            total_size_of_data_in_bytes,
            progress_bar_size=50, # by default
            enable_checks=False
        )
        
        # by definition of a megabyte
        nb_bytes_in_one_megabyte = 1024**2
        
        # converting the sizes from bytes to megabytes (for display purposes only)
        size_of_already_downloaded_data_in_megabytes = float(size_of_already_downloaded_data_in_bytes) / nb_bytes_in_one_megabyte
        total_size_of_data_in_megabytes = float(total_size_of_data_in_bytes) / nb_bytes_in_one_megabyte
        
        # by default
        precision_sizes = 2
        
        str_size_of_already_downloaded_data_in_megabytes = f"{size_of_already_downloaded_data_in_megabytes:.{precision_sizes}f}"
        str_total_size_of_data_in_megabytes = f"{total_size_of_data_in_megabytes:.{precision_sizes}f}"
        
        nb_additional_zeros = len(str_total_size_of_data_in_megabytes) - len(str_size_of_already_downloaded_data_in_megabytes)
        assert nb_additional_zeros >= 0
        if nb_additional_zeros != 0:
            additional_zeros = "0" * nb_additional_zeros
            str_size_of_already_downloaded_data_in_megabytes = additional_zeros + str_size_of_already_downloaded_data_in_megabytes
        
        current_progress_bar += f" Downloaded {str_size_of_already_downloaded_data_in_megabytes}/{str_total_size_of_data_in_megabytes} MB"
        
        clear_currently_printed_row()
        print(current_progress_bar, end="\r")


def _download_data(
        data_URL: str,
        path_of_downloaded_data: str,
        *,
        hash_value: Optional[str] = None
    ) -> None:
    """
    Automatically downloads the data located at the specified URL, and saves
    it to the specified path. If the data already exists on your disk, then
    nothing is done
    
    This function is basically a wrapper of the `urlretrieve` method (of the
    `urllib.request` module)
    
    Naturally, assuming the data doesn't already exist on your disk, an
    internet connection is required to run this function !
    
    For security purposes, you can also input the (SHA-256) hash value of the
    downloaded data (assuming you know it before downloading the data). In that
    case, the hash of the downloaded data will be compared to the specified
    value (and an exception will be raised if both hashes don't match)
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    assert isinstance(data_URL, str)
    assert len(data_URL.strip()) > 0
    data_URL = data_URL.strip()
    
    assert isinstance(path_of_downloaded_data, str)
    assert len(path_of_downloaded_data.strip()) > 0
    path_of_downloaded_data = path_of_downloaded_data.strip()
    
    assert isinstance(hash_value, (type(None), str))
    if hash_value is not None:
        assert len(hash_value.strip()) > 0
        hash_value = hash_value.strip()
        assert len(hash_value) == 64 # specific to the SHA-256 hashing algorithm
    
    # ---------------------------------------------------------------------- #
    
    # if the data already exists on your disk, there is no need to re-download it
    download_is_required = not(os.path.exists(path_of_downloaded_data))
    
    if download_is_required:
        try:
            print(f"\nDownloading the data from the URL \"{data_URL}\". This might take a couple of seconds ...\n")
            t_beginning_downloading = perf_counter()
            
            # actually downloading the raw MNIST data from `data_URL`, and
            # saving it to the location `default_path_of_downloaded_data`
            urlretrieve(
                url=data_URL,
                filename=path_of_downloaded_data,
                reporthook=_download_progress_bar
            )
            
            assert os.path.exists(path_of_downloaded_data)
            
            t_end_downloading = perf_counter()
            duration_downloading = t_end_downloading - t_beginning_downloading
            print(f"\n\nSuccessfully downloaded the data to the location \"{path_of_downloaded_data}\". Done in {duration_downloading:.3f} seconds")
        except (Exception, KeyboardInterrupt):
            print("\n")
            if os.path.exists(path_of_downloaded_data):
                os.remove(path_of_downloaded_data)
            raise
    
    if hash_value is not None:
        # checking if the hash value of the downloaded data is valid or not
        # (for security purposes)
        _validate_hash_of_downloaded_data(
            path_of_downloaded_data,
            hash_value
        )


##############################################################################


# Functions related to the display of the styled confusion matrix


def highlight_diagonal(
        conf_matrix_as_dataframe: DataFrame,
        *,
        color_of_diagonal: str = "green"
    ) -> DataFrame:
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
    assert conf_matrix_as_dataframe.ndim == 2
    nb_classes = conf_matrix_as_dataframe.shape[0]
    assert nb_classes >= 2
    assert conf_matrix_as_dataframe.shape[1] == nb_classes
    conf_matrix_dtype = conf_matrix_as_dataframe.to_numpy().dtype.type
    assert issubclass(conf_matrix_dtype, np.integer)
    
    # checking `color_of_diagonal`
    assert isinstance(color_of_diagonal, str)
    assert len(color_of_diagonal.strip()) > 0
    color_of_diagonal = color_of_diagonal.strip().lower()
    assert " " not in color_of_diagonal
    
    # ---------------------------------------------------------------------- #
    
    diagonal_mask = np.full(conf_matrix_as_dataframe.shape, "", dtype="<U24")
    
    CSS_property = f"background-color: {color_of_diagonal};"
    np.fill_diagonal(diagonal_mask, CSS_property)
    
    diagonal_mask_as_dataframe = DataFrame(
        diagonal_mask,
        index=conf_matrix_as_dataframe.index,
        columns=conf_matrix_as_dataframe.columns
    )
    
    return diagonal_mask_as_dataframe


def highlight_all_cells(
        value: Union[str, float],
        *,
        colormap: Colormap = cm.Greens
    ) -> str:
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
        conf_matrix: np.ndarray,
        *,
        selected_classes: Union[str, list[int]] = "all",
        dict_of_real_class_names: Optional[dict[int, str]] = None,
        normalize: str = "no",
        precision: int = 1,
        color: str = "green",
        offset_spacing: int = 1,
        display_with_line_breaks: bool = False
    ) -> None:
    """
    Prints the styled confusion matrix. The result won't be the same depending
    on whether you're running the program on a Jupyter notebook or not !
    
    Here, `conf_matrix` is a non-normalized confusion matrix (i.e. a raw,
    integer-valued confusion matrix). By design, `conf_matrix` is meant to
    be the output of the `confusion_matrix` function (of the "core.py" script)
    
    The kwarg `selected_classes` can either be :
        - the string "all", if you're working with all the classes (default)
        - a list/tuple/1D-array containing the specific class indices you're
          working with (e.g. [2, 4, 7])
    
    The kwarg `dict_of_real_class_names`, if not set to `None`, is a dictionary
    with :
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
    """
    # ====================================================================== #
    
    # checking the validity of the specified arguments
    
    assert isinstance(conf_matrix, np.ndarray)
    assert conf_matrix.ndim == 2
    nb_classes = conf_matrix.shape[0]
    assert nb_classes >= 2
    assert conf_matrix.shape[1] == nb_classes
    assert issubclass(conf_matrix.dtype.type, np.integer)
    
    selected_classes, class_names = _validate_selected_classes(
        selected_classes,
        nb_classes,
        dict_of_real_class_names=dict_of_real_class_names
    )
    assert len(class_names) == nb_classes
    
    assert isinstance(normalize, str)
    normalize = normalize.strip().lower()
    possible_values_for_normalize_kwarg = ["no", "columns", "rows"]
    if normalize not in possible_values_for_normalize_kwarg:
        raise ValueError(f"get_confusion_matrix_as_dataframe (utils.py) - Unrecognized value for the `normalize` kwarg : \"{normalize}\" (possible values : {list_to_string(possible_values_for_normalize_kwarg)})")
    
    if normalize != "no":
        assert isinstance(precision, int)
        assert precision >= 0
    
    assert isinstance(color, str)
    assert len(color.strip()) > 0
    color = color.strip().lower()
    
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
        raise ValueError(f"print_confusion_matrix (utils.py) - Unrecognized value for the `color` kwarg : \"{color}\" (possible color names : {list_to_string(list(COLORS_AND_COLORMAPS))})")
    
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
                if np.allclose(sum_of_column, 0):
                    normalized_conf_matrix[:, class_index] = 0
                else:
                    normalized_conf_matrix[:, class_index] /= sum_of_column
            
            elif normalize == "rows":
                # computing the RECALL of the (true) class at row index `class_index`
                sum_of_row = np.sum(normalized_conf_matrix[class_index, :])
                if np.allclose(sum_of_row, 0):
                    raise Exception(f"print_confusion_matrix (utils.py) - The true class \"{class_names[class_index]}\" (class_index={class_index}) isn't represented in the confusion matrix !")
                normalized_conf_matrix[class_index, :] /= sum_of_row
        
        # rounding the (rescaled) normalized confusion matrix to the specified
        # precision
        normalized_conf_matrix = np.round(100 * normalized_conf_matrix, precision)
        
        conf_matrix_as_dataframe = DataFrame(
            normalized_conf_matrix,
            index=class_names,
            columns=class_names
        )
        
        for class_name in class_names:
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
    
    assert conf_matrix_as_dataframe.columns.name not in class_names
    assert conf_matrix_as_dataframe.index.name not in class_names
    
    # building the header/description of the confusion matrix
    if normalize == "no":
        conf_matrix_header = "\nRAW CONFUSION MATRIX"
    elif normalize == "columns":
        conf_matrix_header = "\nNETWORK PRECISION - NORMALIZED CONFUSION MATRIX"
    elif normalize == "rows":
        conf_matrix_header = "\nNETWORK RECALL - NORMALIZED CONFUSION MATRIX"
    conf_matrix_header += f" (normalized=\"{normalize}\") :"
    
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
            # to the value they hold (relative to the used colormap), i.e.
            # the returned confusion matrix will have a background gradient
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
        
        print(conf_matrix_header)
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
    
    color_reset = Style.RESET_ALL
    
    lines_of_str_conf_matrix_as_dataframe = str_conf_matrix_as_dataframe.split("\n")
    lines_of_str_colored_conf_matrix_as_dataframe = lines_of_str_conf_matrix_as_dataframe.copy()
    
    max_length_of_class_names = max([len(class_name) for class_name in class_names])
    
    # first index (inclusive) of the area of interest of each row (i.e. the
    # area of each row that potentially contains the actual numeric data of
    # the confusion matrix)
    first_index_area_of_interest = len(offset_spacing) + max(len(conf_matrix_as_dataframe.columns.name), len(conf_matrix_as_dataframe.index.name), max_length_of_class_names)
    
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
            # updating the current column names (for classes)
            
            current_class_names_in_columns = current_row.split()[1 : ]
            if current_class_names_in_columns[-1] == "\\":
                # here, the trailing "\" means that there's a line break
                current_class_names_in_columns = current_class_names_in_columns[ : -1]
            
            continue
        
        # updating the current row name (for classes)
        assert content_of_beginning_of_row in class_names
        current_class_name_in_rows = content_of_beginning_of_row
        
        # ------------------------------------------------------------------ #
        
        # if we made it to this point, then it means that the current row
        # is a row of interest (i.e. it holds the actual numeric data of
        # the confusion matrix)
        
        area_of_interest = current_row[first_index_area_of_interest : ]
        
        split_area_of_interest = area_of_interest.split(" ")
        split_colored_area_of_interest = split_area_of_interest.copy()
        
        # ------------------------------------------------------------------ #
        
        # this counter holds the number of numeric values that have been
        # encountered in the area of interest of the current row
        nb_seen_numeric_values_in_row = 0
        
        current_class_name_in_columns = None
        
        for element_index, element in enumerate(split_area_of_interest):
            # here, `element_is_numeric_value` is a boolean indicating whether
            # `element` is an integer or a float (i.e. a numeric value)
            try:
                _ = float(element)
                element_is_numeric_value = True
            except:
                element_is_numeric_value = False
            
            if element_is_numeric_value:
                nb_seen_numeric_values_in_row += 1
                
                # updating the current column name (for classes)
                current_class_name_in_columns = current_class_names_in_columns[nb_seen_numeric_values_in_row - 1]
            
            if current_class_name_in_columns is None:
                continue
            
            if (current_class_name_in_columns == current_class_name_in_rows) and (element_is_numeric_value or (element == "%")):
                # here, `element` is on the diagonal of the confusion matrix,
                # therefore color will be added to it !
                
                if element_is_numeric_value:
                    try:
                        next_element_is_a_percent_symbol = (split_colored_area_of_interest[element_index + 1] == "%")
                    except:
                        next_element_is_a_percent_symbol = False
                    
                    colored_element = printed_color_of_diagonal + element
                    if not(next_element_is_a_percent_symbol):
                        colored_element += color_reset
                    split_colored_area_of_interest[element_index] = colored_element
                    
                    if not(next_element_is_a_percent_symbol):
                        break
                else:
                    assert element == "%"
                    split_colored_area_of_interest[element_index] += color_reset
                    
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
    
    # Enables the ability to print colored text in the standard output (of Python
    # consoles/terminals). This method is imported from the `colorama` module
    init()
    
    print(conf_matrix_header)
    print(str_colored_conf_matrix_as_dataframe)

