"""
Provide `scores` as an xarray accessor.

Allows for invocation from an xarray object directly.

```python
import scores.experimental

# Get Sample Data
forecast = scores.sample_data.simple_forecast()

# Get MSE from observations
forecast.scores.continuous.mse(scores.sample_data.simple_observations())


```
"""

from types import ModuleType
from typing import Any, Callable, Union

import xarray as xr

xarrayType = Union[xr.DataArray, xr.Dataset]

import scores

class ScoresWrapper:
    """
    Thin Wrapper Around scores packages for use with xarray objects.

    Any `__getattr__` calls will be absorbed by this class to follow the API of `scores`.
    """

    def __init__(self, xarray_obj: xarrayType, function: Union[Callable, None] = None) -> None:
        self._obj = xarray_obj
        
        function = function or scores
        self.function = function 
        

    def help(self):
        """Get help for underlying function"""
        return help(self.function)

    def __getattr__(self, key: str):
        """
        Get attribute.
        
        Intercepts the call to wrap around called functions.
        Passes xarray object, and rebuilds a `ScoresWrapper` to allow for module descent.
        """
        if key == "function":
            raise AttributeError(f"{self} has no attribute {key!r}")

        if not hasattr(self.function, key):
            raise AttributeError(f"{self.function} has no attribute {key!r}")
        new_func = getattr(self.function, key)

        if not isinstance(new_func, (Callable, ModuleType)):
            return new_func

        return self.__class__(self._obj, new_func)

    def __repr__(self):
        name = self.function.__qualname__ if hasattr(self.function, '__qualname__') else self.function.__name__
        return f"xarray wrapper for {name!r}.\nCall `.help()` to see help."

    def __dir__(self):
        return self.function.__dir__()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call underlying function with passed arguments"""
        return self.function(self._obj, *args, **kwargs)


xr.register_dataarray_accessor("scores")(ScoresWrapper)
xr.register_dataset_accessor("scores")(ScoresWrapper)