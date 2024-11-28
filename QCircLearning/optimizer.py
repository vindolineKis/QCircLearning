from typing import Union, Callable, Optional
from scipy.optimize import minimize, OptimizeResult


class Optimizer:

    def __init__(self, method: Optional[Union[str, Callable]] = "BFGS") -> None:
        self.method = method
        self.saved_path = None

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
        method: Optional[Union[str, Callable]] = None,
        **kwargs
    ) -> OptimizeResult:
        if record_path:
            self.path_x, self.path_y = [], []

            def min_func(x):
                self.path_x.append(x)
                y = func(x)
                self.path_y.append(y)
                return y

        else:
            min_func = func

        method = method or self.method
        if callable(method):
            return method(min_func, x0, callback=callback, **kwargs)
        else:
            return minimize(min_func, x0, method=method, callback=callback, options=kwargs)
