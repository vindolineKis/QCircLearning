from typing import Union, Callable, Optional
from scipy.optimize import minimize, OptimizeResult
from cust_optimizer import NNOptimizer, RSOptimizer


class Optimization:

    def __init__(self, method: Optional[Union[str, Callable]] = "BFGS") -> None:
        self.method = method
        self.saved_path = None
    _available_methods = ['Neural Network', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'random search']

    @staticmethod
    def list_methods():
        return Optimization._available_methods
    
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
  
        if method == 'Neural Network':
            method_nn = NNOptimizer() 
            return method_nn.forward_fitting(func=min_func, x0=x0, callback=callback, **kwargs)  
        elif method == 'random search':
            method_rs = RSOptimizer()
            return method_rs.random_search(func=min_func, x0=x0, callback=callback, **kwargs)
        elif method in ['BFGS', 'Nelder-Mead', 'Powell', 'CG']:
            return minimize(min_func, x0, method=method, jac='3-point' if method == 'BFGS' else None, callback=callback, options=kwargs)
        else:
            raise ValueError(f'Optimizer method {method} not available. Available methods are {self.list_methods()}')

