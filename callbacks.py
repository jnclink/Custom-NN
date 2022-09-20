# -*- coding: utf-8 -*-

"""
Script defining some callback classes that can be used during the training loop
"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from utils import list_to_string


##############################################################################


class Callback(ABC):
    """
    Base (abstract) callback class
    """
    def __init__(self) -> None:
        pass
    
    def __str__(self) -> str:
        # default string representation of the callback classes (most of the
        # time their `__str__` method will override this one)
        return f"{self.__class__.__name__}()"
    
    def __repr__(self) -> str:
        return str(self)
    
    @abstractmethod
    def callback(self, *args: object, **kwargs: object) -> object:
        """
        Defines the callback of the current class
        """
        pass


##############################################################################


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback class
    """
    
    # class variable
    MONITORED_VALUES: list[str] = [
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy"
    ]
    
    def __init__(
            self,
            monitor: str = "train_loss",
            *,
            patience: int = 5
        ) -> None:
        
        # checking the validity of the `monitor` kwarg (i.e. the monitored metric)
        assert isinstance(monitor, str)
        assert len(monitor.strip()) > 0
        monitor = monitor.strip().lower().replace(" ", "_")
        if monitor not in EarlyStoppingCallback.MONITORED_VALUES:
            raise ValueError(f"EarlyStoppingCallback.__init__ - Unrecognized value for the `monitor` kwarg : \"{monitor}\" (possible values for this kwarg : {list_to_string(EarlyStoppingCallback.MONITORED_VALUES)})")
        self.monitor: str = monitor
        
        # checking the validity of the `patience` kwarg
        assert isinstance(patience, int)
        assert patience >= 2
        self.patience: int = patience
        
        if self.monitor in ["train_loss", "val_loss"]:
            # the loss needs to be *minimized*, therefore here we'll
            # check if the loss has *only been increasing* for the past
            # `self.patience` epochs
            self.comparison_function: Callable = max
        elif self.monitor in ["train_accuracy", "val_accuracy"]:
            # the accuracy needs to be *maximized*, therefore here we'll
            # check if the accuracy has *only been decreasing* for the past
            # `self.patience` epochs
            self.comparison_function: Callable = min
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(monitor=\"{self.monitor}\", patience={self.patience})"
    
    def callback(
            self,
            history: dict[str, list],
            *,
            enable_checks: bool = True
        ) -> bool:
        """
        Returns a boolean indicating whether the network should prematurely
        stop the training loop at the end of the current epoch or not
        """
        # ------------------------------------------------------------------ #
        
        # checking the validity of the specified arguments
        
        assert isinstance(enable_checks, bool)
        
        if enable_checks:
            assert isinstance(history, dict)
            assert len(history) in [3, 5]
            
            for key, value in history.items():
                assert isinstance(key, str)
                if key != "epoch":
                    assert key in EarlyStoppingCallback.MONITORED_VALUES
                
                assert isinstance(value, list)
                assert len(value) > 0
            
            assert self.monitor in history
        
        # ------------------------------------------------------------------ #
        
        monitored_values = history[self.monitor]
        nb_completed_epochs = len(monitored_values)
        
        if nb_completed_epochs >= self.patience:
            # we're only interest in the last `self.patience` epochs
            values_of_interest = monitored_values[-self.patience : ]
            assert len(values_of_interest) == self.patience # necessary check
            
            # checking if the values of interest form a strictly monotonous
            # sequence or not (the type of the monotony depends on `self.monitor`)
            for index_of_epoch_of_interest in range(self.patience - 1):
                value_of_interest      = values_of_interest[index_of_epoch_of_interest]
                next_value_of_interest = values_of_interest[index_of_epoch_of_interest + 1]
                
                if np.allclose(value_of_interest, next_value_of_interest):
                    return False
                
                # checking if the sequence formed by `value_of_interest` and
                # `next_value_of_interest` is strictly monotonous (in the
                # "wrong direction")
                compared_value = self.comparison_function(value_of_interest, next_value_of_interest)
                if np.allclose(compared_value, value_of_interest):
                    return False
            
            # if we made it to this point, then the sequence defined by the
            # `self.patience` last monitored values is strictly monotonous :
            # it's strictly increasing if a loss is being monitored, or it's
            # strictly decreasing if an accuracy is being monitored
            return True
        
        return False

