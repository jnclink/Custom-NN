# -*- coding: utf-8 -*-

"""
Script defining miscellaneous useful functions
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
        np.float64
    )
    
    if datatype not in AVAILABLE_DATATYPES:
        if isinstance(datatype, type):
            str_datatype = datatype.__name__
        else:
            str_datatype = str(datatype)
        raise ValueError(f"is_an_available_global_datatype (utils.py) - Unrecognized value for `datatype` : \"{str_datatype}\" (has to be float32 or float64)")


# checking the hard-coded value of `DEFAULT_DATATYPE` (just in case there's a typo)
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
    will return the default value of `DEFAULT_DATATYPE`, not the updated one !
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
    Converts the specified non-empty list (or tuple or 1D numpy array) into a string
    """
    if not(isinstance(L, (list, tuple, np.ndarray))):
        raise TypeError(f"list_to_string (utils.py) - The input `L` isn't a list (or a tuple or a 1D numpy array), it's a \"{L.__class__.__name__}\" !")
    
    if isinstance(L, np.ndarray):
        assert len(L.shape) == 1, "The input `L` has to be 1-dimensional !"
    
    nb_elements = len(L)
    assert nb_elements > 0, "The input `L` is empty !"
    
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


def check_if_label_vector_is_valid(y):
    """
    Checks the validity of the label vector `y`
    """
    assert isinstance(y, np.ndarray)
    assert len(y.shape) == 1
    assert issubclass(y.dtype.type, np.integer)
    assert np.min(y) >= 0


##############################################################################


# Functions related to the `mnist_dataset.py` script


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
        - the string "all" (if you want to work with all the digits ranging
          from 0 to 9)
        - a list/tuple/1D-array containing the specific digits you want to work
          with (e.g. [2, 4, 7])
    """
    assert isinstance(dict_of_label_vectors, dict)
    assert precision >= 0
    
    if isinstance(selected_classes, str):
        selected_classes = selected_classes.lower()
        if selected_classes != "all":
            raise ValueError(f"display_distribution_of_classes (utils.py) - Invalid value for the kwarg `selected_classes` : \"{selected_classes}\" (expected : \"all\", or the specific list of selected classes)")
    else:
        if not(isinstance(selected_classes, (list, tuple, np.ndarray))):
            raise TypeError(f"display_distribution_of_classes (utils.py) - The `selected_classes` kwarg isn't a string, list, tuple or a 1D numpy array, it's a \"{selected_classes.__class__.__name__}\" !")
        
        if not(isinstance(selected_classes, np.ndarray)):
            selected_classes = np.array(selected_classes)
        check_if_label_vector_is_valid(selected_classes)
        
        distinct_selected_classes = np.unique(selected_classes)
        nb_distinct_selected_classes = distinct_selected_classes.size
        assert nb_distinct_selected_classes >= 2, "display_distribution_of_classes (utils.py) - Please select at least 2 distinct classes in the `selected_classes` kwarg !"
    
    displayed_string = "\nClass distributions :\n"
    
    for label_vector_name, label_vector in dict_of_label_vectors.items():
        assert isinstance(label_vector_name, str)
        check_if_label_vector_is_valid(label_vector)
        
        displayed_string += f"\n{label_vector_name} :"
        
        nb_labels = label_vector.size
        nb_classes = np.unique(label_vector).size
        assert nb_classes >= 2
        
        if isinstance(selected_classes, str):
            # here, `selected_classes` is equal to the string "all"
            classes = np.arange(nb_classes)
        else:
            assert nb_classes == nb_distinct_selected_classes
            classes = distinct_selected_classes
        
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


def to_categorical(y, dtype=int):
    """
    Performs one-hot encoding on a 1D vector of INTEGER labels
    """
    check_if_label_vector_is_valid(y)
    
    nb_labels = y.size
    
    distinct_labels, distinct_labels_inverse = np.unique(y, return_inverse=True)
    nb_classes = distinct_labels.size
    assert nb_classes >= 2
    
    y_categorical = np.zeros((nb_labels, nb_classes), dtype=dtype)
    
    for label_index, label in enumerate(distinct_labels_inverse):
        # by definition
        y_categorical[label_index, label] = 1
    
    return y_categorical


def categorical_to_vector(y):
    """
    Converts a 2D categorical matrix (one-hot encoded matrix or logits) into
    its associated 1D vector of INTEGER labels
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
    encoded equivalent (in that case, `labels` will be a 2D matrix)
    """
    
    batches = {
        "data"   : [],
        "labels" : []
    }
    
    if len(labels.shape) == 1:
        check_if_label_vector_is_valid(labels)
    
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
    check_if_label_vector_is_valid(y_true)
    check_if_label_vector_is_valid(y_pred)
    assert y_true.size == y_pred.size
    
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
    
    NB : If you decide to use the `confusion_matrix` function of the
          `sklearn.metrics` module, just be aware that the output of their
          function is the TRANSPOSED version of the "common" definition of the
          confusion matrix (i.e. the transposed version of the output of this
          function)
    """
    check_if_label_vector_is_valid(y_true)
    check_if_label_vector_is_valid(y_pred)
    assert y_true.size == y_pred.size
    
    nb_classes = np.unique(y_true).size
    assert nb_classes >= 2
    
    conf_matrix = np.zeros((nb_classes, nb_classes), dtype=int)
    
    for predicted_class, actual_class in zip(y_pred, y_true):
        # by definition (rows=predicted_classes and columns=actual_classes)
        conf_matrix[predicted_class, actual_class] += 1
    
    return conf_matrix


def print_confusion_matrix(
        conf_matrix,
        selected_classes="all",
        normalize="rows",
        precision=1,
        initial_spacing=1,
        display_with_line_breaks=True,
        jupyter_notebook=False
    ):
    """
    Prints the confusion matrix in a more user-friendly way
    
    Here, `conf_matrix` is a non-normalized confusion matrix (i.e. a raw,
    integer-valued confusion matrix)
    
    The kwarg `selected_classes` can either be :
        - the string "all" (if you want to work with all the digits ranging
          from 0 to 9)
        - a list/tuple/1D-array containing the specific digits you want to work
          with (e.g. [2, 4, 7])
    """
    # checking the validity of the `normalize` kwarg
    assert isinstance(normalize, str)
    normalize = normalize.lower()
    possible_values_for_normalize_kwarg = ["rows", "columns", "no"]
    if normalize not in possible_values_for_normalize_kwarg:
        raise ValueError(f"get_confusion_matrix_as_dataframe (utils.py) - Unrecognized value for the `normalize` kwarg : \"{normalize}\" (possible values : {list_to_string(possible_values_for_normalize_kwarg)})")
    
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
    
    # Only keeping the selected classes in the confusion matrix (if specified)
    
    nb_classes = conf_matrix.shape[0]
    assert nb_classes >= 2
    
    if isinstance(selected_classes, str):
        selected_classes = selected_classes.lower()
        if selected_classes != "all":
            raise ValueError(f"print_confusion_matrix (utils.py) - Invalid value for the kwarg `selected_classes` : \"{selected_classes}\" (expected : \"all\", or the specific list of selected classes)")
        
        class_names = [str(digit) for digit in range(nb_classes)]
    else:
        if not(isinstance(selected_classes, (list, tuple, np.ndarray))):
            raise TypeError(f"print_confusion_matrix (utils.py) - The `selected_classes` kwarg isn't a string, list, tuple or a 1D numpy array, it's a \"{selected_classes.__class__.__name__}\" !")
        
        if not(isinstance(selected_classes, np.ndarray)):
            selected_classes = np.array(selected_classes)
        check_if_label_vector_is_valid(selected_classes)
        
        distinct_selected_classes = np.unique(selected_classes)
        nb_distinct_selected_classes = distinct_selected_classes.size
        assert nb_distinct_selected_classes == nb_classes, f"print_confusion_matrix (utils.py) - The number of distinct classes in the `selected_classes` kwarg (= {nb_distinct_selected_classes}) doesn't match the shape of `conf_matrix` (= {conf_matrix.shape}) !"
        
        class_names = [str(digit) for digit in distinct_selected_classes]
    
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
        conf_matrix_as_dataframe = pd.DataFrame(normalized_conf_matrix, index=class_names, columns=class_names)
        
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


# Function related to the `layers.py` script (it's used by the `__str__` methods
# of both the `ActivationLayer` and `DropoutLayer` classes)


def count_nb_decimals_places(x, max_precision=6):
    """
    Returns the number of decimal places of the scalar `x`
    """
    assert np.isscalar(x)
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
        return count_nb_decimals_places(x - int(x))

