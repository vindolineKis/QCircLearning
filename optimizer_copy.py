from mimetypes import init
from typing import List
from scipy.optimize import minimize, OptimizeResult
from scipy.special import softmax   
from nn_trainer import trainer_model
import numpy as np
from sklearn.linear_model import LinearRegression
  

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
            return minimize(min_func, x0, method=self._NN_opt, jac='3-point',callback=callback, options=kwargs)
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
        # flush the output
        sys.stdout.flush()

        for i in range(len(nn_models)):
            nn_models[i].summary()
        
        # adam_pitimizer = adamTrainer()

        for _ in range(max_iter):
            res.nit += 1

            for model in nn_models:
                model.compile(optimizer='adam', 
                                loss='mse',
                                metrics=[], )
                


                fit_his = model.fit(sample_x,
                            sample_y,
                            epochs=classcial_epochs,
                            verbose=verbose)
                
                x0 = optimal[0] + np.random.normal(0, .02, para_size)
                # print("x0",x0.shape)
                
                prediction0 = model.back_minimize(x0=x0,method='L-BFGS-B', verbose=verbose)
                # prediction0_res = adam_pitimizer.minimization(init_para=x0,
                #                                             target_func=func,
                #                                             target_func_div= model.get_gradient2,
                #                                             verbose=verbose,
                #                                             callback=None,
                #                                             disp=True,
                #                                             stop=True,
                #                                             other_args=None)

                # prediction0 = prediction0_res.x
                # save these points to the path
                if np.linalg.norm(prediction0 - x0) > 1e-3:
                    print(f'Prediction is different from x0: {np.linalg.norm(prediction0 - x0)}')
                # Evaluate on real quantum computer
                y0 = func(prediction0)
                # print(f'data size ({model.name}):', len(sample_x))
                sys.stdout.flush()
                if y0 < optimal[1]:
                    optimal = [prediction0, y0]
                sample_x = np.append(sample_x, [prediction0], axis=0)
                sample_y = np.append(sample_y, y0)

                # training from 1 random points
                y_vec = sample_y
                # print(f'y_vec:{y_vec}')
                y_gs = softmax(-100*y_vec)
                
                # random sample 4 points from the index by the probability
                index = np.random.choice(len(y_gs), 2, replace=True, p=y_gs)
                # print(index.shape)
                
                # add 3 points to the training data
                for i in index:
                    # x1 = sample_x[i]
                    x1 = sample_x[i] + np.random.normal(0, .02, para_size)
                    prediction1 = model.back_minimize(x0=x1,method='L-BFGS-B', verbose=verbose)
                    # prediction1 = prediction1.x
                    # save these points to the path
                    y1 = func(prediction1)

                    # record the difference when prediction is different from x0
                    # if np.linalg.norm(prediction1 - x1) > 1e-3:
                    #     print(f'Prediction is different from x0: {np.linalg.norm(prediction1 - x1)}')
                    
                    # Evaluate on real quantum computer
                    if y1 < optimal[1]:
                        optimal = [x1, y1]

                    sample_x = np.append(sample_x, [prediction1], axis=0)
                    sample_y = np.append(sample_y, y1)
                    print(f'training from randome select points by index {index} ')
                    sys.stdout.flush()
                
                index1 = np.random.choice(len(y_gs), 2, replace=True, p=y_gs)

                for i in index1:
                    x2 = sample_x[i] + np.random.normal(0, .02, para_size)
                    # prediction2 = model.back_minimize(x0=x2,method='L-BFGS-B', verbose=verbose)
                
                    # save these points to the path
                    y2 = func(x2)

                    # record the difference when prediction is different from x0
                    # if np.linalg.norm(prediction2 - x2) > 1e-3:
                    #     print(f'Prediction is different from x0: {np.linalg.norm(prediction2 - x2)}')
                    
                    # Evaluate on real quantum computer
                    if y2 < optimal[1]:
                        optimal = [x2, y2]

                    sample_x = np.append(sample_x, [x2], axis=0)
                    sample_y = np.append(sample_y, y2)
                    print(f'training from randome select points by index {index1} ')
                    sys.stdout.flush()
            res.nfev += 1
        
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
    
