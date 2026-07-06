from typing import Union, Callable, Optional
from scipy.optimize import minimize, OptimizeResult
import numpy as np

class Optimizer:

    def __init__(self, method: Optional[Union[str, Callable]] = "BFGS") -> None:
        self.method = method
        self.saved_path = None
        self.path_x = []
        self.path_y = []
        self.method_used = None

    @property
    def get_path_x(self):
        return getattr(self, "path_x", None)

    @property
    def get_path_y(self):
        return getattr(self, "path_y", None)

    def optimize(
        self,
        func,
        x0,
        callback=None,
        record_path: bool = True,
        override_method: Optional[Union[str, Callable]] = None,
        scipy_options: Optional[dict] = None,
       
        **kwargs
    ) -> OptimizeResult:
        # reset path_x and path_y for each optimization run
        self.path_x = []
        self.path_y = []

        if record_path:
            def min_func(x):
                x_copy = np.array(x).copy() # Ensure x is a numpy array
                y = func(x_copy)
                
                self.path_x.append(x_copy)
                self.path_y.append(y)
                return y

        else:
            min_func = func

        method_used = override_method if override_method is not None else self.method
        self.method_used = method_used

        if callable(method_used):
            return method_used(
                min_func,
                np.array(x0).copy(),
                callback=callback,
                **kwargs,
            )

        else:
            return minimize(
                min_func,
                np.array(x0).copy(),
                method=method_used,
                callback=callback,
                options=scipy_options,
            )