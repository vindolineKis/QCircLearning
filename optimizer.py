from mimetypes import init
from typing import List
from scipy.optimize import minimize, OptimizeResult
# from scipy.special import softmax   
from nn_trainer import trainer_model
import numpy as np
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping
import itertools

import sys

class Optimizer:

    _available_methods = ['Neural Network', 'Nelder-Mead', 'Powell', 'CG', 'BFGS','linear_regression','random search','Adam']

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
            def min_func(x):
                return func(x)

        if method is None:
            method = self.method

        if method not in Optimizer._available_methods:
            raise ValueError(f'Optimizer method {method} not available. Available methods are {self.list_methods()}')
        
        # options = {k: v for k, v in kwargs.items() if k not in ['init_data', 'max_iter', 'classcial_epochs', 'verbose', 'NN_Models']}
        # args = kwargs.get('args', ())

        if method == 'Neural Network':
            return minimize(min_func, x0, method=self._NN_opt, jac='3-point', callback=callback, options=kwargs)
        # elif method == 'linear_regression':
        #     return minimize(min_func, x0, method=self._linear_model_opt, callback=callback, options=kwargs)
        elif method == 'random search':
            return minimize(min_func, x0, method=self._random_search, callback=callback, options=kwargs)
        elif method == 'BFGS':
            return minimize(min_func, x0, method=method,jac='3-point', callback=callback)
        else:
            return minimize(min_func, x0, method=method, callback=callback)

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

        if 'classical_epochs' in kwargs:
            classcial_epochs:int = kwargs['classical_epochs']
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
        # flush the output
        sys.stdout.flush()
        # # train from the diffirent initial point
        # for i in range(len(sample_y)):
        # # train from the diffirent initial point
        #     for model in nn_models:
        #         model.compile(optimizer='adam', 
        #                         loss='mse',
        #                         metrics=[], )
                
        #         fit_his = model.fit(sample_x,
        #                     sample_y,
        #                     epochs=classcial_epochs,
        #                     verbose=verbose)
        #         x0 = sample_x[i]
        #         prediction0 = model.back_minimize(x0=x0,method='L-BFGS-B', verbose=verbose)
        #         y = func(prediction0)
        #         if y < optimal[1]:
        #             optimal = [prediction0, y]    
                    
        for model in nn_models:
            model.summary()
        # flush the output
        sys.stdout.flush()
        for _ in range(max_iter):
            res.nit += 1
            
            for model in nn_models:
                
                model.compile(optimizer='adam', 
                                loss='mse',
                                metrics=[], )
                early_stop = EarlyStopping(monitor='loss', patience=5, verbose=verbose)
                
                fit_his = model.fit(sample_x,
                            sample_y,
                            epochs=classcial_epochs,
                            batch_size=64,
                            verbose=verbose,
                            callbacks=[early_stop])
                # print the training history
                # print(fit_his.history)

                x0 = optimal[0] + np.random.normal(0, 0.02, para_size)
                sys.stdout.flush()
            
                prediction0 = model.back_minimize(x0=x0,method='L-BFGS-B', verbose=verbose)

                if np.linalg.norm(prediction0 - x0) > 1e-3:
                    print(f'Prediction is different from x0: {np.linalg.norm(prediction0 - x0)}')
                
                # else:
                #     print(f'Prediction is too close to x0: {np.linalg.norm(prediction0 - x0)}')
                #     x0 = np.random.uniform(-np.pi/2,np.pi/2, len(x0)) 
                #     prediction0 = x0
                # Evaluate on real quantum computer
                y0 = func(prediction0)
                
                # print(f'data size ({model.name}):', len(sample_x))
                sys.stdout.flush()
                if y0 < optimal[1]:
                    optimal = [prediction0, y0]
                # sample_x = np.append(sample_x, [prediction0], axis=0)
                # sample_y = np.append(sample_y, y0)

                # Gather all prediction variations (original, +2π, and -2π)
                offset =[0, np.pi*2, -np.pi*2]

                predictions = np.array(list(itertools.product(*[[x + o for o in offset] for x in prediction0])))
                # predictions = np.array(list(itertools.product(*[[x + o for o in offset] for x in prediction0])))
                # predictions = np.vstack([prediction0, prediction0+ np.pi*2,prediction0 - np.pi*2])
                                
                sample_x = np.concatenate([sample_x, predictions], axis=0)
                # Extend sample_y with the corresponding y0 values (same for each variation)
                sample_y = np.concatenate([sample_y, [y0] * len(predictions)])
                print(f'data size ({model.name}):', len(sample_x))

                res.nfev += 1
                # random points for the next point
                if np.abs(y0-optimal[1])< 1e-5 and optimal[1]>1e-4:
                    for i in range(2):
                        print(f'Cost is too close to optimal: {np.abs(y0-optimal[1])}')
                        x0 = np.random.uniform(-np.pi,np.pi, len(x0)) 
                        y0 = func(x0)
                        if y0 < optimal[1]:
                            optimal = [x0, y0]
                        sample_x = np.append(sample_x, [x0], axis=0)
                        sample_y = np.append(sample_y, y0)
    
        res.x = np.copy(optimal[0])
        res.fun = np.copy(optimal[1])
    
        return res
    

    
    def _random_search(self,func, x0, callback=None, **kwargs):
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


        sample_x = init_data
        sample_y = np.array([])

        optimal = [None,1] # [opt_para, opt_y]

        for para in sample_x: # generate the initial points

            y = func(para)

            if y < optimal[1]:
                optimal = [para, y]
            sample_y = np.append(sample_y, y)

        print(f'Training with  random search')
        # flush the output
        sys.stdout.flush()

        

        for _ in range(max_iter):
            res.nit += 1

            x0 = optimal[0] + np.random.normal(0, .02, para_size)

            
            prediction = x0

            # Evaluate on real quantum computer
            y = func(prediction)
            res.nfev += 1
            sys.stdout.flush()

            if y < optimal[1]:
                optimal = [prediction, y]
            sample_x = np.append(sample_x, [prediction], axis=0)
            sample_y = np.append(sample_y, y)
        
        res.x = np.copy(optimal[0])
        res.fun = np.copy(optimal[1])
        # res.message = message
        # res.nit = i + 1
        return res
    


