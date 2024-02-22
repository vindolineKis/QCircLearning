from mimetypes import init
from typing import List
from scipy.optimize import minimize, OptimizeResult
from nn_trainer import trainer_model
import numpy as np

class Optimizer:

    _available_methods = ['Neural Network', 'Nelder-Mead', 'Powell', 'CG', 'BFGS']

    @staticmethod
    def list_methods():
        return Optimizer._available_methods

    def __init__(self, method:str='Neural Network') -> None:
        self.method = method
        
        self.saved_path = None
    
    @property
    def get_path_x(self):
        if self.saved_path is None:
            return None
        else:
            return self.path_x
    @property
    def get_path_y(self):
        if self.saved_path is None:
            return None
        else:
            return self.path_y

    def optimize(self,
                 func,
                 x0,
                 callback=None,
                 record_path:bool=True,
                 method:str=None,
                 **kwargs) -> OptimizeResult:

        if record_path:
            self.path_x = []
            self.path_y = []
            def min_func(x):
                self.path_x.append(x)
                y = func(x)
                self.path_y.append(y)
                return y
        else:
            min_func = func

        if method is None:
            method = self.method

        if method not in Optimizer._available_methods:
            raise ValueError(f'Optimizer method {method} not available. Available methods are {self.list_methods()}')
        
        if method == 'Neural Network':
            return minimize(min_func, x0, method=self._NN_opt, callback=callback, options=kwargs)
        else:
            return minimize(min_func, x0, method=method,callback=callback)

    def _NN_opt(self,func, x0, callback=None, **kwargs):
        # optimize using neural network
        para_size = len(x0)
        res = OptimizeResult()
        res.nfev = 0
        res.nit = 0

        # Define the default values
        if 'init_data' in kwargs:
            init_data:List = kwargs['init_data']
        else:
            init_data = np.random.uniform(-10,10,(60,para_size))

        if 'max_iter' in kwargs:
            max_iter:int = kwargs['max_iter']
        else:
            max_iter:int = 20

        if 'classcial_epochs' in kwargs:
            classcial_epochs:int = kwargs['classcial_epochs']
        else:
            classcial_epochs:int = 20
        
        if 'verbose' in kwargs:
            verbose:int = kwargs['verbose']
        else:
            verbose:int = 0
        
        if 'NN_Models' in kwargs:
            nn_models:list = kwargs['NN_Models']
        else:
            nn_models:list = [
                trainer_model.default_model((para_size,)),
                trainer_model.simple_model((para_size,)),
            ]

        sample_x = init_data
        sample_y = np.array([])

        optimal = [None,1] # [opt_para, opt_y]

        for para in sample_x: # generate the initial points

            y = func(para)

            if y < optimal[1]:
                optimal = [para, y]
            sample_y = np.append(sample_y, y)

        print(f'Training with the neural networks')
        nn_models[0].summary()
        nn_models[1].summary()

        for _ in range(max_iter):
            res.nit += 1

            for model in nn_models:
                model.compile(optimizer='adam',
                                loss='mse',
                                metrics=[])

                fit_his = model.fit(sample_x,
                            sample_y,
                            epochs=classcial_epochs,
                            verbose=verbose)
                x0 = optimal[0] + np.random.normal(0, .02, para_size)
                prediction = model.back_minimize(x0=x0,method='BFGS', verbose=verbose)

                # Evaluate on real quantum computer
                y = func(prediction)
                res.nfev += 1

                # FIXME: Debug
                print(f'debug ({model.name}):', res.nfev, fit_his.history['loss'][-1], y)

                if y < optimal[1]:
                    optimal = [prediction, y]
                sample_x = np.append(sample_x, [prediction], axis=0)
                sample_y = np.append(sample_y, y)
        
        res.x = np.copy(optimal[0])
        res.fun = np.copy(optimal[1])
        # res.message = message
        # res.nit = i + 1
        return res
