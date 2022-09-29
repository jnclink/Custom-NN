# -*- coding: utf-8 -*-

"""
Script defining some regularizers
"""

from __future__ import annotations

import numpy as np

from utils import count_nb_decimals_places


##############################################################################


class Regularizer:
    """
    Base regularizer class (created for structural purposes only)
    
    The regularizer classes are only used as "decorative classes"
    """
    def __init__(self) -> None:
        pass
    
    def __str__(self) -> str:
        # default string representation of the regularizer classes (most of
        # the time their `__str__` method will override this one)
        return f"{self.__class__.__name__}()"
    
    def __repr__(self) -> str:
        return str(self)


##############################################################################


class L1(Regularizer):
    """
    L1 regularizer class
    """
    def __init__(self, L1_coeff: float) -> None:
        assert isinstance(L1_coeff, float)
        assert (L1_coeff > 0) and (L1_coeff < 1)
        self.L1_coeff = L1_coeff
    
    def __str__(self) -> str:
        L1_coeff_precision = max(2, count_nb_decimals_places(self.L1_coeff))
        str_L1_coeff = f"{self.L1_coeff:.{L1_coeff_precision}f}"
        
        return f"{self.__class__.__name__}(L1_coeff={str_L1_coeff})"
    
    def __eq__(self, obj: object) -> bool:
        if type(obj) != type(self):
            return False
        return np.allclose(obj.L1_coeff, self.L1_coeff)


##############################################################################


class L2(Regularizer):
    """
    L2 regularizer class (it basically has the same structure as L1, but with
    a different name)
    """
    def __init__(self, L2_coeff: float) -> None:
        assert isinstance(L2_coeff, float)
        assert (L2_coeff > 0) and (L2_coeff < 1)
        self.L2_coeff = L2_coeff
    
    def __str__(self) -> str:
        L2_coeff_precision = max(2, count_nb_decimals_places(self.L2_coeff))
        str_L2_coeff = f"{self.L2_coeff:.{L2_coeff_precision}f}"
        
        return f"{self.__class__.__name__}(L2_coeff={str_L2_coeff})"
    
    def __eq__(self, obj: object) -> bool:
        if type(obj) != type(self):
            return False
        return np.allclose(obj.L2_coeff, self.L2_coeff)


##############################################################################


class L1_L2(Regularizer):
    """
    Class merging both L1 and L2 regularizer classes
    """
    def __init__(self, *args: float) -> None:
        assert len(args) in [1, 2]
        
        if len(args) == 1:
            common_L1_L2_coeff = args[0]
            L1_coeff, L2_coeff = common_L1_L2_coeff, common_L1_L2_coeff
        elif len(args) == 2:
            L1_coeff, L2_coeff = args
        
        L1.__init__(self, L1_coeff)
        L2.__init__(self, L2_coeff)
    
    def __str__(self) -> str:
        L1_coeff_precision = max(2, count_nb_decimals_places(self.L1_coeff))
        str_L1_coeff = f"{self.L1_coeff:.{L1_coeff_precision}f}"
        
        L2_coeff_precision = max(2, count_nb_decimals_places(self.L2_coeff))
        str_L2_coeff = f"{self.L2_coeff:.{L2_coeff_precision}f}"
        
        return f"{self.__class__.__name__}(L1_coeff={str_L1_coeff}, L2_coeff={str_L2_coeff})"
    
    def __eq__(self, obj: object) -> bool:
        if type(obj) != type(self):
            return False
        return np.allclose(obj.L1_coeff, self.L1_coeff) and np.allclose(obj.L2_coeff, self.L2_coeff)

