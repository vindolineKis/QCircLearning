from mimetypes import init
from typing import List
from scipy.optimize import minimize, OptimizeResult
# from scipy.special import softmax   
from nn_trainer import trainer_model
import numpy as np
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
  

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
             record_path: bool = True,
             method: str = None,
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

        # For Neural Network, directly call _NN_opt instead of using minimize
        if method == 'Neural Network':
            return self._NN_opt(func=min_func, x0=x0, callback=callback, **kwargs)

        elif method == 'random search':
            return self._random_search(func=min_func, x0=x0, callback=callback, **kwargs)

        elif method in ['BFGS', 'Nelder-Mead', 'Powell', 'CG']:
            return minimize(min_func, x0, method=method, jac='3-point' if method == 'BFGS' else None, callback=callback, options=kwargs)

        else:
            raise ValueError(f'Unknown optimization method: {method}')


    def _NN_opt(self,func, x0, callback=None, **kwargs):
        # optimize using neural network
        para_size = len(x0)
        res = OptimizeResult()
        res.nfev = 0
        res.nit = 0

        # Define the default values
        init_data = kwargs.get('init_data', np.random.uniform(-10, 10, (60, para_size)))
        max_iter = kwargs.get('max_iter', 20)
        classical_epochs = kwargs.get('classical_epochs', 20)
        batch_size = kwargs.get('batch_size', 16)
        verbose = kwargs.get('verbose', 0)
        nn_models = kwargs.get('NN_Models', [
        trainer_model.default_model((para_size,)),
        trainer_model.simple_model((para_size,)),
    ])


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
        
                    
        for model in nn_models:
            model.summary()
        # flush the output
            sys.stdout.flush()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("early stopping")
            early_stopping = EarlyStopping(
                monitor='loss',
                min_delta=1e-12,
                patience=50,  # Stop training if no improvement in 5 epochs
                mode='min',
                restore_best_weights=True
            )
            print("reduce_lr")
            reduce_lr = ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1
            )
          
            for iteration in range(max_iter):
                res.nit += 1
                
                for model in nn_models:
                    
                    model.compile(optimizer=Adam(learning_rate=1e-3), 
                                    loss='mse',

                                    metrics=[], )
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(classical_epochs)
                    if len(sample_x) < batch_size:
                        epochs = classical_epochs
                    else:
                        epochs = max(classical_epochs - (len(sample_x) // batch_size + 1), 1)  # Ensure at least 1 epoch

                    fit_his = model.fit(sample_x,
                                sample_y,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=verbose,
                                callbacks=[early_stopping, reduce_lr])
                    # print the training history
                    # Print loss and epoch informatio
                    print("====================================================")
                    print(f"Iteration {iteration + 1}/{max_iter}")
                    print(f'epochs: {epochs}')
                    print(f'sample size: {len(sample_x)}')
                    for epoch, loss in enumerate(fit_his.history['loss'], start=1):
                        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.8f}")
                    sys.stdout.flush()
                    print("====================================================")
                   
                    x0 = optimal[0] 
                    # + np.random.normal(0, 0.02, para_size)
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
        
                    # Gather all prediction variations (original, +4π, and -4π)
                    predictions = np.vstack([prediction0, prediction0+ np.pi*2,prediction0 - np.pi*2])
                    # Extend sample_x with the new predictions
                    # f_1 = func(predictions[1])
                    # f_2 = func(predictions[2])
                    # # check if the y0,f_1,f_2 are the same
                    # allclose = np.allclose([y0,f_1,f_2], y0, rtol=1e-5)
                    # if allclose:
                    #     print(f'True True True: {y0}, {f_1}, {f_2}')
                    # else:
                    #     print(f'False False False: {y0}, {f_1}, {f_2}')
                    
                                    
                    sample_x = np.concatenate([sample_x, predictions], axis=0)
                    # Extend sample_y with the corresponding y0 values (same for each variation)
                    sample_y = np.concatenate([sample_y, [y0] * 3])

                    res.nfev += 1
                    # random points for the next point
                    if np.abs(y0-optimal[1])< 1e-5 and optimal[1]>1e-3:
                        for i in range(2):
                            print(f'Cost is too close to optimal: {np.abs(y0-optimal[1])}')
                            x0 = np.random.uniform(-np.pi*2,np.pi*2, len(x0)) 
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
    


