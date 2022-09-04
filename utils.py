# -*- coding: utf-8 -*-

"""
Miscellaneous useful functions
"""

import numpy as np
import pandas as pd


##############################################################################


# Defining the datatype of ALL the data that will flow through the network
# (i.e. we're defining the "global datatype")


# global variables (these are the 2 only global variables of the entire project)
DEFAULT_DATATYPE = "float32"
DTYPE_RESOLUTION = np.finfo(DEFAULT_DATATYPE).resolution


def is_an_available_global_datatype(datatype):
    """
    Checks if the specified datatype can be the "global datatype" of the
    network (or not). For now, the only available datatypes are float32 and float64
    """
    if isinstance(datatype, str):
        datatype = datatype.lower().replace(" ", "")
    
    AVAILABLE_DATATYPES = (
        "float32",
        np.float32,
        "float64",
        np.float64,
        np.float_   # is either np.float32 or np.float64, depending on your PC
    )
    
    if datatype not in AVAILABLE_DATATYPES:
        if isinstance(datatype, type):
            str_datatype = datatype.__name__
        else:
            str_datatype = str(datatype)
        raise ValueError(f"is_an_available_global_datatype (utils.py) - Unrecognized value for `datatype` : \"{str_datatype}\" (has to be float32 or float64)")


# checking the hard-coded value of `DEFAULT_DATATYPE` (just in case)
is_an_available_global_datatype(DEFAULT_DATATYPE)


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
    will be the default value of `DEFAULT_DATATYPE`, not the updated one !
    The same goes with the global variable `DTYPE_RESOLUTION`
    """
    if datatype == float:
        set_global_datatype(np.float_)
    
    # checking the specified datatype first
    is_an_available_global_datatype(datatype)
    
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
    if np.isscalar(x):
        assert type(x) == np.dtype(dtype)
    else:
        # here, `x` is a vector/matrix
        assert x.dtype == dtype


def cast(x, dtype):
    """
    Returns the cast version of `x` to `dtype`. Here, `x` can either be a
    scalar or a vector/matrix. By design, in most cases, `dtype` will be
    equal to `utils.DEFAULT_DATATYPE`
    """
    if np.isscalar(x):
        # there might be a more efficient way to do this, but at least it works
        cast_x = np.array([x], dtype=dtype)[0]
    else:
        # here, `x` is a vector/matrix
        cast_x = x.astype(dtype)
    
    check_dtype(cast_x, dtype)
    return cast_x


##############################################################################


# Functions related to the `mnist_dataset.py` script


def get_type_of_array(array):
    """
    Returns the type of an array (as a string)
    """
    assert isinstance(array, np.ndarray)
    return "numpy." + str(array.dtype)


def get_range_of_array(array, precision=3):
    """
    Returns the range of an array (as a string)
    """
    assert isinstance(array, np.ndarray)
    min_element_of_array = np.min(array)
    max_element_of_array = np.max(array)
    
    FLOAT_TYPES = (
        float,
        np.float_,
        np.floating,
        np.float16,
        np.float32,
        np.float64
    )
    
    INTEGER_TYPES = (
        int,
        np.int_,
        np.integer,
        np.int8,  np.uint8,
        np.int16, np.uint16,
        np.int32, np.uint32,
        np.int64, np.uint64
    )
    
    array_dtype = array.dtype.type
    
    if array_dtype in FLOAT_TYPES:
        # float-like data
        str_range = f"{min_element_of_array:.{precision}f} -> {max_element_of_array:.{precision}f}"
    else:
        # integer-like data
        assert array_dtype in INTEGER_TYPES
        str_range = f"{min_element_of_array} -> {max_element_of_array}"
    return str_range


def display_distribution_of_classes(dict_of_label_vectors, precision=2):
    """
    Prints the distribution of classes in the specified label vectors.
    The input, `dict_of_label_vectors`, is a dictionary containing :
        - as its keys   : the names of the label vectors (as strings)
        - as its values : the corresponding 1D vectors of INTEGER labels
    """
    displayed_string = "\nClass distributions :\n"
    
    for label_vector_name, label_vector in dict_of_label_vectors.items():
        assert isinstance(label_vector_name, str)
        assert isinstance(label_vector, np.ndarray)
        
        displayed_string += f"\n{label_vector_name} :"
        
        nb_labels = label_vector.size
        nb_classes = np.unique(label_vector).size
        
        for digit in range(nb_classes):
            nb_corresponding_digits = np.where(label_vector == digit)[0].size
            proportion = 100 * float(nb_corresponding_digits) / nb_labels
            str_proportion = f"{proportion:.{precision}f}"
            if proportion < 10:
                str_proportion = "0" + str_proportion
            displayed_string += f"\n    {digit} --> {str_proportion} %"
    
    print(displayed_string)


##############################################################################


# Functions used to switch from an integer vector of labels to a one-hot encoded
# matrix (and vice-versa)


def to_categorical(y, dtype=int):
    """
    Performs one-hot encoding on a 1D vector of INTEGER labels
    """
    assert len(y.shape) == 1
    
    nb_labels = y.size
    
    distinct_elements_of_y = np.unique(y)
    nb_classes = distinct_elements_of_y.size
    
    # checking that all the integers between 0 (inclusive) and `nb_classes - 1`
    # (inclusive) are contained in the vector `y`
    for class_index in range(nb_classes):
        assert class_index in distinct_elements_of_y
    
    y_categorical = np.zeros((nb_labels, nb_classes), dtype=dtype)
    
    for label_index in range(nb_labels):
        # by definition
        y_categorical[label_index, y[label_index]] = 1
    
    return y_categorical


def categorical_to_vector(y):
    """
    Converts a one-hot encoded (2D) matrix into the associated 1D vector of
    INTEGER labels
    """
    assert len(y.shape) == 2
    return np.argmax(y, axis=1)


##############################################################################


# Main function used to split the input data into batches


def split_data_into_batches(
        data,
        labels,
        batch_size,
        normalize_batches=True,
        nb_shuffles=10,
        seed=None
    ):
    """
    Splits the input data and labels into batches with `batch_size` samples each
    (if `batch_size` doesn't divide the number of samples, then the very last
    batch will simply have `nb_samples % batch_size` samples)
    
    Here, `labels` can either be a 1D vector of INTEGER labels or its one-hot
    encoded equivalent (in that case it will be a 2D matrix)
    """
    
    batches = {
        "data"   : [],
        "labels" : []
    }
    
    nb_samples = data.shape[0]
    assert labels.shape[0] == nb_samples
    assert batch_size <= nb_samples
    
    batch_indices = np.arange(nb_samples)
    
    # shuffling the batch indices
    np.random.seed(seed)
    for shuffle_index in range(nb_shuffles):
        np.random.shuffle(batch_indices)
    np.random.seed(None) # resetting the seed
    
    if normalize_batches:
        # here we're assuming that each sample does NOT have a standard
        # deviation equal to zero (i.e. we're assuming that there are at
        # least 2 different pixel values in each sample)
        normalized_data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
    
    for batch_index in range(0, nb_samples, batch_size):
        indices_of_current_batch =  batch_indices[batch_index : batch_index + batch_size]
        
        # checking if the batch size is correct
        if indices_of_current_batch.size != batch_size:
            # if the batch size isn't equal to `batch_size`, then it means
            # that we are generating the very last batch (it also implies that
            # `batch_size` doesn't divide `nb_samples`)
            assert batch_index == nb_samples - nb_samples % batch_size
            assert indices_of_current_batch.size > 0
            assert indices_of_current_batch.size == nb_samples % batch_size
        
        if normalize_batches:
            data_current_batch = normalized_data[indices_of_current_batch, :]
        else:
            data_current_batch = data[indices_of_current_batch, :]
        labels_current_batch = labels[indices_of_current_batch]
        
        batches["data"].append(data_current_batch)
        batches["labels"].append(labels_current_batch)
    
    assert np.vstack(tuple(batches["data"])).shape == data.shape
    
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
    assert len(batches["labels"]) == expected_nb_batches
    
    return batches


##############################################################################


# Functions related to the accuracy metric


def accuracy_score(y_true, y_pred, normalize=True):
    """
    Returns the proportion of the correctly predicted samples. The returned
    proportion lies between 0 and 1
    
    Here, `y_true` and `y_pred` are 1D vectors of INTEGER labels
    """
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 1
    
    acc_score = np.where(y_true == y_pred)[0].size
    if normalize:
        nb_test_samples = y_true.size
        acc_score = float(acc_score) / nb_test_samples
    return acc_score


def confusion_matrix(y_true, y_pred):
    """
    Returns the raw confusion matrix of the tuple (y_true, y_pred). Its shape
    will be (nb_classes, nb_classes), and, for all integers `i` and `j` in the
    range [0, nb_classes - 1], the value of `conf_matrix[i, j]` (say, for
    instance, `N`) indicates that :
        - Out of all the test samples that were predicted to belong to class `i`,
          `N` of them actually belonged to class `j`
        - Or, equivalently, out of all the test samples that actually belonged
          to class `j`, `N` of them were predicted to belong to class `i`
    
    Here, `y_true` and `y_pred` are 1D vectors of INTEGER labels
    
    NB : If you decide to use the `confusion_matrix` function of the
          `sklearn.metrics` module, just be aware that the output of their
          function is the TRANSPOSED version of the "common" definition of the
          confusion matrix (i.e. the transposed version of the output of this
          function)
    """
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 1
    
    nb_classes = np.unique(y_true).size
    conf_matrix = np.zeros((nb_classes, nb_classes), dtype=int)
    
    nb_test_samples = y_true.size
    for test_sample_index in range(nb_test_samples):
        # by definition (rows=predicted_classes and columns=actual_classes)
        conf_matrix[y_pred[test_sample_index], y_true[test_sample_index]] += 1
    
    return conf_matrix


def print_confusion_matrix(
        conf_matrix,
        normalize="rows",
        precision=1,
        initial_spacing=1,
        display_with_line_breaks=True,
        jupyter_notebook=False
    ):
    """
    Prints the confusion matrix in a more user-friendly way
    
    Here, `conf_matrix` is a non-normalized confusion matrix
    """
    
    assert isinstance(normalize, str)
    normalize = normalize.lower()
    if normalize not in ["rows", "columns", "no"]:
        raise ValueError(f"get_confusion_matrix_as_dataframe (utils.py) - Unrecognized value for the `normalize` kwarg : \"{normalize}\"")
    
    assert precision >= 0
    if not(jupyter_notebook):
        assert initial_spacing >= 0
    
    pd.options.mode.chained_assignment = None
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("expand_frame_repr", display_with_line_breaks)
    
    # ---------------------------------------------------------------------- #
    
    # converting the confusion matrix into its associated dataframe
    
    nb_classes = conf_matrix.shape[0]
    class_names = [str(digit) for digit in range(nb_classes)]
    
    if normalize != "no":
        normalized_conf_matrix = np.float_(np.copy(conf_matrix))
        for class_index in range(nb_classes):
            if normalize == "rows":
                # computing the PRECISION of the (predicted) class `class_index`
                sum_of_row = np.sum(normalized_conf_matrix[class_index, :])
                if sum_of_row == 0:
                    normalized_conf_matrix[class_index, :] = 0
                else:
                    normalized_conf_matrix[class_index, :] /= sum_of_row
            elif normalize == "columns":
                # computing the RECALL of the (actual) class `class_index` (note
                # that `sum_of_column` cannot be equal to zero, otherwise the number
                # of classes would be strictly less than `nb_classes`, which is
                # absurd)
                sum_of_column = np.sum(normalized_conf_matrix[:, class_index])
                normalized_conf_matrix[:, class_index] /= sum_of_column
        normalized_conf_matrix = np.round(100 * normalized_conf_matrix, precision)
        conf_matrix_as_dataframe = pd.DataFrame(normalized_conf_matrix, index=class_names, columns=class_names)
        for class_name in class_names:
            conf_matrix_as_dataframe[class_name] = conf_matrix_as_dataframe[class_name].astype(str) + " %"
            if not(jupyter_notebook):
                # highlighting the diagonal values with vertical bars
                diagonal_percentage = conf_matrix_as_dataframe[class_name][int(class_name)]
                conf_matrix_as_dataframe[class_name][int(class_name)] = f"|| {diagonal_percentage} ||"
    else:
        conf_matrix_as_dataframe = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    
    # by definition
    conf_matrix_as_dataframe.columns.name = "ACTUAL"
    conf_matrix_as_dataframe.index.name   = "PREDICTED"
    
    print(f"\nCONFUSION MATRIX (normalized=\"{normalize}\") :")
    
    # ---------------------------------------------------------------------- #
    
    # highlighting the diagonal in green (on Jupyter notebook only)
    
    if jupyter_notebook:
        def highlight_diagonal(dataframe, background_color="green"):
            attribute = f"background-color: {background_color};"
            dataframe_mask = np.full(dataframe.shape, "", dtype="<U24")
            np.fill_diagonal(dataframe_mask, attribute)
            dataframe_mask = pd.DataFrame(dataframe_mask, index=dataframe.index, columns=dataframe.columns)
            return dataframe_mask
        
        # `conf_matrix_styler` is a Pandas Styler object (`pandas.io.formats.style.Styler`)
        conf_matrix_styler = conf_matrix_as_dataframe.style.apply(
            highlight_diagonal,
            axis=None
        )
        return conf_matrix_styler
    
    # ---------------------------------------------------------------------- #
    
    # adding the initial spacing
    initial_spacing = " " * initial_spacing
    str_conf_matrix_as_dataframe = f"\n{initial_spacing}" + str(conf_matrix_as_dataframe).replace("\n", f"\n{initial_spacing}") + "\n"
    
    print(str_conf_matrix_as_dataframe)


##############################################################################


# Function related to the `layers.py` script (it's used by some of the
# `__str__` methods of the layer classes)


def count_nb_decimals_places(x, max_precision=6):
    """
    Returns the number of decimal places of the scalar `x`
    """
    assert np.isscalar(x)
    
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
        return count_nb_decimals_places(x - int(x))

