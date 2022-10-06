# -*- coding: utf-8 -*-

"""
Script containing some core functions of the project
"""

from __future__ import annotations

from typing import Union, Optional, Iterator

import numpy as np

from utils import (
    basic_split,
    _validate_split_data_into_batches_inputs,
    _validate_label_vector,
    _validate_numpy_dtype
)


##############################################################################


# Main functions used to split the input data into batches


def split_data_into_batches_as_generator_function(
        data: np.ndarray,
        batch_size: int,
        *,
        labels: Optional[np.ndarray] = None,
        nb_shuffles: int = 10,
        seed: Optional[int] = None,
        enable_checks: bool = True
    ) -> Iterator[np.ndarray]:
    """
    Generator function counterpart of the `split_data_into_batches` function
    of this script. This function has to be defined separately, as the Python
    interpreter automatically considers any function that contains the `yield`
    keyword to be a generator function
    
    Splits the input data and/or labels into batches with `batch_size` samples
    each. If `batch_size` doesn't divide the number of samples, then the very
    last batch will simply have `nb_samples % batch_size` samples !
    
    Here, if `labels` is not equal to `None`, it can either be a 1D vector of
    INTEGER labels or its one-hot encoded equivalent (in that case, `labels`
    will be a 2D matrix)
    
    This function will essentially be used for the training and testing data
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_split_data_into_batches_inputs(
            data,
            batch_size,
            labels,
            nb_shuffles,
            seed
        )
    
    # ---------------------------------------------------------------------- #
    
    # initialization
    
    _has_labels = (labels is not None)
    
    nb_samples = data.shape[0]
    batch_indices = np.arange(nb_samples)
    
    if nb_shuffles > 0:
        # shuffling the batch indices
        np.random.seed(seed)
        for shuffle_index in range(nb_shuffles):
            np.random.shuffle(batch_indices)
        np.random.seed(None) # resetting the seed
    
    # ---------------------------------------------------------------------- #
    
    # actually splitting the data and/or labels into batches
    
    for first_index_of_batch in range(0, nb_samples, batch_size):
        last_index_of_batch = first_index_of_batch + batch_size
        
        indices_of_current_batch = batch_indices[first_index_of_batch : last_index_of_batch]
        
        # checking if the batch size is correct (it's a necessary check)
        if indices_of_current_batch.size != batch_size:
            # if the batch size isn't equal to `batch_size`, then it means
            # that we are generating the very last batch (it also implies that
            # `batch_size` doesn't divide `nb_samples`)
            very_last_batch_size = nb_samples % batch_size
            assert first_index_of_batch == nb_samples - very_last_batch_size
            assert very_last_batch_size > 0
            assert indices_of_current_batch.size == very_last_batch_size
        
        data_current_batch = data[indices_of_current_batch, :]
        
        if not(_has_labels):
            yield data_current_batch
        else:
            labels_current_batch = labels[indices_of_current_batch]
            yield data_current_batch, labels_current_batch


def split_data_into_batches(
        data: np.ndarray,
        batch_size: int,
        *,
        labels: Optional[np.ndarray] = None,
        is_generator: bool = False,
        nb_shuffles: int = 10,
        seed: Optional[int] = None,
        enable_checks: bool = True
    ) -> Union[dict[str, list[np.ndarray]], Iterator[np.ndarray]]:
    """
    Splits the input data and/or labels into batches with `batch_size` samples
    each. If `batch_size` doesn't divide the number of samples, then the very
    last batch will simply have `nb_samples % batch_size` samples !
    
    Here, if `labels` is not equal to `None`, it can either be a 1D vector of
    INTEGER labels or its one-hot encoded equivalent (in that case, `labels`
    will be a 2D matrix)
    
    The "non-generator" form of this function will essentially only be used
    for the validation data
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_split_data_into_batches_inputs(
            data,
            batch_size,
            labels,
            nb_shuffles,
            seed
        )
        assert isinstance(is_generator, bool)
    
    # ---------------------------------------------------------------------- #
    
    # returning the batch generator (if requested)
    
    if is_generator:
        batch_generator = split_data_into_batches_as_generator_function(
            data,
            batch_size,
            labels=labels,
            nb_shuffles=nb_shuffles,
            seed=seed,
            enable_checks=False
        )
        return batch_generator
    
    # ---------------------------------------------------------------------- #
    
    # initialization
    
    _has_labels = (labels is not None)
    
    batches = {
        "data" : []
    }
    if _has_labels:
        batches["labels"] = []
    
    nb_samples = data.shape[0]
    batch_indices = np.arange(nb_samples)
    
    if nb_shuffles > 0:
        # shuffling the batch indices
        np.random.seed(seed)
        for shuffle_index in range(nb_shuffles):
            np.random.shuffle(batch_indices)
        np.random.seed(None) # resetting the seed
    
    # ---------------------------------------------------------------------- #
    
    # actually splitting the data and/or labels into batches
    
    for first_index_of_batch in range(0, nb_samples, batch_size):
        last_index_of_batch = first_index_of_batch + batch_size
        
        indices_of_current_batch = batch_indices[first_index_of_batch : last_index_of_batch]
        
        # checking if the batch size is correct (it's a necessary check)
        if indices_of_current_batch.size != batch_size:
            # if the batch size isn't equal to `batch_size`, then it means
            # that we are generating the very last batch (it also implies that
            # `batch_size` doesn't divide `nb_samples`)
            very_last_batch_size = nb_samples % batch_size
            assert first_index_of_batch == nb_samples - very_last_batch_size
            assert very_last_batch_size > 0
            assert indices_of_current_batch.size == very_last_batch_size
        
        data_current_batch = data[indices_of_current_batch, :]
        if _has_labels:
            labels_current_batch = labels[indices_of_current_batch]
        
        batches["data"].append(data_current_batch)
        if _has_labels:
            batches["labels"].append(labels_current_batch)
    
    # ---------------------------------------------------------------------- #
    
    # checking if the resulting batches are valid or not (before
    # returning them)
    
    if enable_checks:
        expected_nb_batches = (nb_samples + batch_size - 1) // batch_size
        
        assert np.vstack(tuple(batches["data"])).shape == data.shape
        assert len(batches["data"]) == expected_nb_batches
        
        if _has_labels:
            if labels.ndim == 1:
                # in this case, the labels are a 1D vector of INTEGER values
                stacking_function = np.hstack
            elif labels.ndim == 2:
                # in this case, the labels are one-hot encoded (2D matrix)
                stacking_function = np.vstack
            
            assert stacking_function(tuple(batches["labels"])).shape == labels.shape
            assert len(batches["labels"]) == expected_nb_batches
    
    # ---------------------------------------------------------------------- #
    
    return batches


##############################################################################


# Functions related to the accuracy metric (they will basically return the
# same output as their counterparts of the `sklearn.metrics` module)


def accuracy_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        normalize: bool = True,
        enable_checks: bool = True
    ) -> Union[float, int]:
    """
    Returns the proportion of the correctly predicted samples. The returned
    proportion lies between 0 and 1 (if `normalize` is set to `True`)
    
    Here, `y_true` and `y_pred` are 1D vectors of INTEGER labels
    """
    # ---------------------------------------------------------------------- #
    
    # checking the validity of the specified arguments
    
    assert isinstance(enable_checks, bool)
    
    if enable_checks:
        _validate_label_vector(y_true, is_whole_label_vector=False)
        _validate_label_vector(y_pred, is_whole_label_vector=False)
        assert y_true.size == y_pred.size
        
        assert isinstance(normalize, bool)
    
    # ---------------------------------------------------------------------- #
    
    acc_score = int(np.where(y_true == y_pred)[0].size) # we want it to be an `int`, not a `np.int_`
    
    if normalize:
        nb_samples = y_true.size
        acc_score = float(acc_score) / nb_samples
    
    return acc_score


def confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
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
    _validate_label_vector(y_pred, is_whole_label_vector=False)
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


# Main function used to split the raw MNIST data into train/test or
# train/val/test subsets


def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        *,
        test_size:  Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        stratify: bool = False,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ) -> tuple[np.ndarray]:
    """
    Splits data and label arrays into random train and test subsets
    
    Here, `X` has to be a 2D array with numeric values, and `y` has to be a
    1D vector of INTEGER labels
    
    If the `stratify` kwarg is set to `True`, then the resulting `y_train` and
    `y_test` will have roughly the same class distribution as the specified `y`
    
    Sidenote
    --------
    This function will basically do the same task as the associated
    `train_test_split` function of the `sklearn.model_selection` module. The
    only major difference is that, here, the `stratify` kwarg is defined as
    a boolean, whereas it's defined as an array in the original function. Also,
    the reason why the previous `sklearn.model_selection.train_test_split`
    method isn't used is simply because we do NOT want to import Scikit-Learn
    (or "sklearn") ! Indeed, in my opinion, it would be kind of awkward to
    import Scikit-Learn in a project aiming to implement a Deep Learning
    model *from scratch* !
    """
    # ---------------------------------------------------------------------- #
    
    # Checking the specified arguments
    
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    _validate_numpy_dtype(X.dtype) # checking if the data is numeric
    nb_samples = int(X.shape[0])
    
    _validate_label_vector(y)
    assert y.size == nb_samples
    distinct_classes = list(np.unique(y))
    nb_classes = int(len(distinct_classes))
    
    assert isinstance(test_size,  (type(None), float, int))
    assert isinstance(train_size, (type(None), float, int))
    
    if test_size is None:
        if train_size is not None:
            if isinstance(train_size, float):
                test_size = 1 - train_size
            elif isinstance(train_size, int):
                test_size = nb_samples - train_size
        else:
            # default value
            test_size = 0.25
    
    if isinstance(test_size, float):
        assert (test_size > 0) and (test_size < 1)
        nb_test_samples = int(round(test_size * nb_samples))
    elif isinstance(test_size, int):
        nb_test_samples = test_size
    assert nb_test_samples >= nb_classes
    
    if train_size is None:
        assert test_size is not None
        if isinstance(test_size, float):
            train_size = 1 - test_size
        elif isinstance(test_size, int):
            train_size = nb_samples - test_size
    
    if isinstance(train_size, float):
        assert (train_size > 0) and (train_size < 1)
        nb_train_samples = int(round(train_size * nb_samples))
    elif isinstance(train_size, int):
        nb_train_samples = train_size
    assert nb_train_samples >= nb_classes
    
    assert nb_train_samples + nb_test_samples <= nb_samples
    
    assert isinstance(stratify, bool)
    
    assert isinstance(shuffle, bool)
    
    assert isinstance(random_state, (type(None), int))
    if random_state is not None:
        assert random_state >= 0
    
    # ---------------------------------------------------------------------- #
    
    # Basic split
    
    if not(stratify):
        sample_indices = np.arange(nb_samples)
        
        train_indices, test_indices = basic_split(
            sample_indices,
            nb_train_samples,
            nb_test_samples,
            seed=random_state
        )
        
        if shuffle and (random_state is not None):
            # updating the random state (i.e. the "seed") such that the
            # "randomness" of the shuffle is different than the one used in
            # the previous basic split (just in case)
            random_state += 1
    
    # ---------------------------------------------------------------------- #
    
    # Stratified split (i.e. the resulting `y_train` and `y_test` will have
    # roughly the same class distribution as the specified `y`)
    
    else:
        # initializing the train/test indices
        train_indices = []
        test_indices  = []
        
        # keys   : class indices
        # values : (
        #     proportion of the associated class indices (in `y`),
        #     actual index locations (in `y`) of ALL the associated samples
        # )
        class_distribution = {}
        
        # list that will contain the proportions of each class (their order
        # will be determined by the keys of `class_distribution`, i.e. by
        # `distinct_classes`)
        class_proportions = []
        
        # filling `class_distribution` and `class_proportions`
        for class_index in distinct_classes:
            associated_sample_indices = np.where(y == class_index)[0]
            class_proportion = float(associated_sample_indices.size) / nb_samples
            
            class_distribution[class_index] = (class_proportion, associated_sample_indices)
            class_proportions.append(class_proportion)
        
        # 1D array containing the indices of the class proportions, but sorted
        # in ASCENDING order
        sorted_class_proportions = np.argsort(class_proportions)
        
        # sorting the `class_distribution` dictionary by its class proportions
        # (the resulting sorted dictionary is `sorted_class_distribution`)
        sorted_class_distribution = {}
        for index_of_sorted_class in sorted_class_proportions:
            sorted_class = distinct_classes[index_of_sorted_class]
            sorted_class_distribution[sorted_class] = class_distribution[sorted_class]
        
        cumulative_sum_nb_train_samples = 0
        cumulative_sum_nb_test_samples  = 0
        
        nb_processed_classes = 0
        
        for class_proportion, associated_sample_indices in sorted_class_distribution.values():
            # NB : The current class index is the associated key of the dictionary
            #      we're currently iterating over (i.e. `sorted_class_distribution`)
            
            # Here, the current class index will be represented by :
            #     - `partial_nb_train_samples` train samples
            #     - `partial_nb_test_samples` test samples
            
            partial_nb_samples = associated_sample_indices.size
            relative_proportion_of_train_samples = float(nb_train_samples) / (nb_train_samples + nb_test_samples)
            
            if nb_processed_classes != nb_classes - 1:
                partial_nb_train_samples = max(1, int(round(class_proportion * nb_train_samples)))
                partial_nb_test_samples  = max(1, int(round(class_proportion * nb_test_samples)))
                
                # potentially correcting the partial number of train/test
                # samples, due to (small) rounding errors
                excess_nb_samples = (partial_nb_train_samples + partial_nb_test_samples) - partial_nb_samples
                if excess_nb_samples > 0:
                    nb_partial_train_samples_to_remove = int(round(relative_proportion_of_train_samples * excess_nb_samples))
                    nb_partial_test_samples_to_remove  = excess_nb_samples - nb_partial_train_samples_to_remove
                    
                    partial_nb_train_samples = max(1, partial_nb_train_samples - nb_partial_train_samples_to_remove)
                    partial_nb_test_samples  = max(1, partial_nb_test_samples  - nb_partial_test_samples_to_remove)
                
                cumulative_sum_nb_train_samples += partial_nb_train_samples
                cumulative_sum_nb_test_samples  += partial_nb_test_samples
            else:
                # in this case, we've reached the very last class index, which,
                # by design, is also the most represented in `y` (since the
                # `sorted_class_distribution` dictionary is sorted such that
                # the class proportions are in ascending order)
                
                partial_nb_train_samples = nb_train_samples - cumulative_sum_nb_train_samples
                assert partial_nb_train_samples > 0
                
                partial_nb_test_samples = nb_test_samples - cumulative_sum_nb_test_samples
                assert partial_nb_test_samples > 0
            
            assert partial_nb_train_samples + partial_nb_test_samples <= partial_nb_samples
            
            # Here :
            #     - `partial_train_indices` is an array containing the actual
            #        index locations (in `y`) of the randomly selected train
            #        samples associated with the current class index
            #     - `partial_test_indices` is an array containing the actual
            #        index locations (in `y`) of the randomly selected test
            #        samples associated with the current class index
            partial_train_indices, partial_test_indices = basic_split(
                associated_sample_indices,
                partial_nb_train_samples,
                partial_nb_test_samples,
                seed=random_state
            )
            
            # updating the current train/test indices
            train_indices.append(partial_train_indices)
            test_indices.append(partial_test_indices)
            
            nb_processed_classes += 1
            
            if random_state is not None:
                # updating the random state (i.e. the "seed") such that the
                # "randomness" of the basic splits is different for each class
                # index (just in case)
                random_state += 1
        
        assert nb_processed_classes == nb_classes
        
        # concatenating the resulting arrays of train/test indices
        train_indices = np.hstack(tuple(train_indices))
        test_indices  = np.hstack(tuple(test_indices))
    
    # ---------------------------------------------------------------------- #
    
    # Shuffling the train/test sample indices
    
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        np.random.seed(None) # resetting the seed
    
    # ---------------------------------------------------------------------- #
    
    # Actually building the split data
    
    assert train_indices.size == nb_train_samples
    assert test_indices.size == nb_test_samples
    
    X_train = X[train_indices, :].copy()
    X_test  = X[test_indices, :].copy()
    y_train = y[train_indices].copy()
    y_test  = y[test_indices].copy()
    
    assert np.allclose(np.unique(y_train), distinct_classes)
    assert np.allclose(np.unique(y_test),  distinct_classes)
    
    # ---------------------------------------------------------------------- #
    
    return X_train, X_test, y_train, y_test

