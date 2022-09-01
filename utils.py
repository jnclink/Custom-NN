# -*- coding: utf-8 -*-

"""
Miscellaneous useful functions
"""

import numpy as np
import pandas as pd


##############################################################################


def get_type_of_array(array):
    return "numpy." + array.dtype.type.__name__


def get_range_of_array(array, precision=3):
    min_element_of_array = np.min(array)
    max_element_of_array = np.max(array)
    if issubclass(array.dtype.type, float) or (array.dtype.type == np.float32):
        # float-like data
        str_range = f"{min_element_of_array:.{precision}f} -> {max_element_of_array:.{precision}f}"
    else:
        # int-like data
        assert issubclass(array.dtype.type, np.integer)
        str_range = f"{min_element_of_array} -> {max_element_of_array}"
    return str_range


def display_distribution_of_classes(dict_of_label_vectors):
    displayed_string = "\nClass distributions :\n"
    
    for label_vector_name, label_vector in dict_of_label_vectors.items():
        displayed_string += f"\n{label_vector_name} :"
        
        nb_labels = label_vector.size
        nb_classes = np.unique(label_vector).size
        
        for digit in range(nb_classes):
            nb_corresponding_digits = np.where(label_vector == digit)[0].size
            proportion = 100 * float(nb_corresponding_digits) / nb_labels
            str_proportion = f"{proportion:.2f}"
            if proportion < 10:
                str_proportion = "0" + str_proportion
            displayed_string += f"\n    {digit} --> {str_proportion} %"
    
    print(displayed_string)


##############################################################################


def to_categorical(y, dtype="float32"):
    """
    Performs one-hot encoding on a 1D vector of INTEGER labels
    """
    assert len(y.shape) == 1
    
    nb_labels = y.size
    nb_classes = np.unique(y).size
    y_categorical = np.zeros((nb_labels, nb_classes)).astype(dtype)
    
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
    """
    
    batches = {
        "data"   : [],
        "labels" : []
    }
    
    nb_samples = data.shape[0]
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
            assert indices_of_current_batch.size > 0
            assert batch_index == nb_samples - nb_samples % batch_size
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
        stacking_function = np.hstack
    else:
        assert len(labels.shape) == 2
        stacking_function = np.vstack
    assert stacking_function(tuple(batches["labels"])).shape == labels.shape
    
    expected_nb_batches = (nb_samples + batch_size - 1) // batch_size
    assert len(batches["data"]) == expected_nb_batches
    assert len(batches["labels"]) == expected_nb_batches
    
    return batches


##############################################################################


def accuracy_score(y_true, y_pred, normalize=True):
    """
    Here, `y_true` and `y_pred` are 1D vectors of INTEGER labels
    """
    acc_score = np.where(y_true == y_pred)[0].size
    if normalize:
        nb_test_samples = y_true.size
        acc_score = float(acc_score) / nb_test_samples
    return acc_score


def confusion_matrix(y_true, y_pred):
    """
    Here, `y_true` and `y_pred` are 1D vectors of INTEGER labels
    
    NB : If you decide to use the `confusion_matrix` function of the
          `sklearn.metrics` module, just be aware that the output of their
          function is the TRANSPOSED version of the "common" definition of the
          confusion matrix (i.e. the transposed version of the output of this
          function)
    """
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

