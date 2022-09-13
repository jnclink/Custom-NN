# -*- coding: utf-8 -*-

"""
Script containing some core functions of the project
"""

import numpy as np

from utils import _validate_label_vector


##############################################################################


# Main function used to split the input data into batches


def split_data_into_batches(
        data,
        batch_size,
        labels=None,
        nb_shuffles=10,
        seed=None
    ):
    """
    Splits the input data and/or labels into batches with `batch_size` samples
    each. If `batch_size` doesn't divide the number of samples, then the very
    last batch will simply have `nb_samples % batch_size` samples !
    
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
        
        data_current_batch = data[indices_of_current_batch, :]
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


# Functions related to the accuracy metric (they will basically return the
# same output of the associated functions of the `sklearn.metrics` module)


def accuracy_score(y_true, y_pred, normalize=True):
    """
    Returns the proportion of the correctly predicted samples. The returned
    proportion lies between 0 and 1
    
    Here, `y_true` and `y_pred` are 1D vectors of INTEGER labels
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    _validate_label_vector(y_true, is_whole_label_vector=False)
    _validate_label_vector(y_pred, is_whole_label_vector=False)
    assert y_true.size == y_pred.size
    
    assert isinstance(normalize, bool)
    
    # ---------------------------------------------------------------------- #
    
    acc_score = np.where(y_true == y_pred)[0].size
    
    if normalize:
        nb_samples = y_true.size
        acc_score = float(acc_score) / nb_samples
    
    return acc_score


def confusion_matrix(y_true, y_pred):
    """
    Returns the raw confusion matrix of `y_true` and `y_pred`. Its shape will
    be `(nb_classes, nb_classes)`, and, for all integers `i` and `j` in the
    range [0, nb_classes - 1], the value of `conf_matrix[i, j]` (say, for
    instance, `N`) indicates that :
        - Out of all the test samples that were predicted to belong to class `j`,
          `N` of them actually belonged to class `i`
        - Or, equivalently, out of all the test samples that actually belonged
          to class `i`, `N` of them were predicted to belong to class `j`
    
    Here, `y_true` and `y_pred` are 1D vectors of INTEGER labels
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    _validate_label_vector(y_true)
    _validate_label_vector(y_pred)
    assert y_true.size == y_pred.size
    
    # ---------------------------------------------------------------------- #
    
    nb_classes = np.unique(y_true).size
    conf_matrix = np.zeros((nb_classes, nb_classes), dtype=int)
    
    for actual_class, predicted_class in zip(y_true, y_pred):
        # by definition (the rows are the true classes and the columns are
        # the predicted classes)
        conf_matrix[actual_class, predicted_class] += 1
    
    return conf_matrix


##############################################################################


# Main function used to split the raw MNIST data into train/test or train/val/test
# subsets (it will basically to the same task as the associated `train_test_split`
# function of the `sklearn.model_selection` module)


# TODO : Re-code the `train_test_split` method of the `sklearn.model_selection`
#        module (WITH the `stratify` kwarg)

