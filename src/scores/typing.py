"""
This module contains various compound or union types which can be used across the codebase to ensure
a consistent approach to typing is handled.
"""
from collections.abc import Hashable, Iterable
from typing import Any, Optional, Union

import warnings

import pandas as pd
import xarray as xr

# Flexible Dimension Types should be used for preserve_dims and reduce_dims in all
# cases across the repository
FlexibleDimensionTypes = Optional[Iterable[Hashable]]

# Xarraylike data types should be used for all forecast, observed and weights
# However currently some are specified as DataArray only
XarrayLike = Union[xr.DataArray, xr.Dataset]

# These type hint values *may* be used for various arguments across the
# scores repository but are not establishing a standard or expectation beyond
# the function they are used in

FlexibleArrayType = Union[XarrayLike, pd.Series]


# Warning of incorrect types
class TypeWarning(Warning):
    """Warning for types being not what was expected."""


def warn_type(obj: Any, valid_types: Union[tuple[type, ...], type]) -> bool:
    """Check if given `obj` is of the `valid_types`. Raises warning and returns False if not.
    
    Accepts either a type or tuple of types, and can support Union's from `typing`.
    
    Examples:
        >>> warn_type([1], list)
        True
        
        >>> warn_type([1], (tuple, list))
        True
        
        >>> warn_type([1], (tuple,))
        False
        
        >>> from typing import Union
        >>> warn_type([1], Union[tuple, list])
        True
    """

    from typing import _UnionGenericAlias

    if isinstance(valid_types, _UnionGenericAlias):
        valid_types = valid_types.__args__

    if not isinstance(obj, valid_types):
        warnings.warn(f"Expected object of type/s {valid_types!r}, but got {type(obj)}.", TypeWarning)
        return False
    return True