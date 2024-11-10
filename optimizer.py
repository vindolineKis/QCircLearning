from mimetypes import init
from typing import List
from scipy.optimize import minimize, OptimizeResult
# from scipy.special import softmax   
# from scipy.special import softmax   
from nn_trainer import trainer_model
import numpy as np
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import tensorflow as tf
# import tensorflow_addons as tfa

import sys
import os
import random

# Set the seed value for reproducibility
SEED = 42

# Set the PYTHONHASHSEED environment variable
os.environ['PYTHONHASHSEED'] = str(SEED)

# Set random seed for Python's `random` module
random.seed(SEED)

# Set random seed for NumPy
np.random.seed(SEED)

# Set random seed for TensorFlow
tf.random.set_seed(SEED)

# Additional configuration for GPU determinism in TensorFlow
# Make sure operations involving cuDNN are deterministic
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'





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

      


        sample_y = np.array([])
        sample_x = np.array([]).reshape(0,para_size)
        # boundry for the parameters
        # bound_low = -2 * np.pi
        # bound_high = 2 * np.pi
        optimal = [None,1] # [opt_para, opt_y]

        for para in init_data: # generate the initial points

            y = func(para)
            if y < optimal[1]:
                optimal = [para, y]
        
            # prid_data = np.vstack([para, para+ np.pi*2,para - np.pi*2])
            # prid_data = np.clip(prid_data, bound_low, bound_high)
            # len_x = prid_data.shape[0]
            # sample_x = np.vstack([sample_x, prid_data])
            # sample_y = np.append(sample_y, [y]*len_x)
            sample_x = np.append(sample_x, [para], axis=0)
            sample_y = np.append(sample_y, y)
          

        print(f'Training with the neural networks')
        # flush the output
        sys.stdout.flush()
        
                    
        for model in nn_models:
            model.summary()
        # flush the output
            sys.stdout.flush()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print("early stopping")
            # early_stopping = EarlyStopping(
            #     monitor='loss',
            #     # min_delta=1e-6,
            #     patience=100,  # Stop training if no improvement in 5 epochs
            #     mode='min',
            #     restore_best_weights=True
            # )
            # print("reduce_lr")
            # reduce_lr = ReduceLROnPlateau(
            #     monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1
            # )
            # clr = tfa.optimizers.CyclicalLearningRate(
            #     initial_learning_rate=1e-4,
            #     maximal_learning_rate=1e-2,
            #     step_size=2000,
            #     scale_fn=lambda x: 1/(2.**(x-1)),
            #     scale_mode="cycle",
            # )

            for iteration in range(max_iter):

              
                res.nit += 1
                
                for model in nn_models:
                    
                    model.compile(optimizer=Adam(learning_rate=1e-4), 
                                    loss='mse', metrics=['mae'])
                    
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"epochs: {classical_epochs}")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"batch size: {batch_size}")
                    
                    fit_his = model.fit(sample_x,
                                sample_y,
                                epochs=classical_epochs,
                                batch_size=batch_size,
                                verbose=verbose,
                                # callbacks=[early_stopping]
                                )
                    # print the training history
                    # Print loss and epoch informatio
                    print("====================================================")
                    print(f"Iteration {iteration + 1}/{max_iter}")
                    print(f"batch")
                    print(f'epochs: {classical_epochs}')
                    print(f'sample size: {len(sample_x)}')
                    for epoch, loss in enumerate(fit_his.history['loss'], start=1):
                        print(f"Epoch {epoch}/{classical_epochs} - Loss: {loss:.1e}")
                    sys.stdout.flush()
                    print("====================================================")
                   



                    x0 = optimal[0] 
                    # + np.random.normal(0, 0.02, para_size)
                    
                    sys.stdout.flush()
                    
                    prediction0 = model.back_minimize(x0=x0,method='L-BFGS-B', verbose=verbose)
                    # print(f'ite:::::::::::::::::::::::::::::::::::::::: {res.nit}') 

                    # if np.linalg.norm(prediction0 - x0) > 1e-3:
                    #     print(f'Prediction is different from x0: {np.linalg.norm(prediction0 - x0)}')
                    
                    # else:
                    #     print(f'Prediction is too close to x0: {np.linalg.norm(prediction0 - x0)}')
                    #     x0 = np.random.uniform(-np.pi/2,np.pi/2, len(x0)) 
                    #     prediction0 = x0
                    # Evaluate on real quantum computer
                    y0 = func(prediction0)
                    res.nfev += 1
                    # if np.abs(y0)> 1e-4:
                    #     yr = func(x0)
                    #     if yr < y0:
                    #         print(f'Random point is better than prediction: {yr} < {y0}')
                    #         prediction0 = x0
                    #         y0 = yr
                    # print(f'data size ({model.name}):', len(sample_x))
                    # sys.stdout.flush()
                    if y0 < optimal[1]:
                        optimal = [prediction0, y0]
                    # sample_x = np.append(sample_x, [prediction0], axis=0)
                    # sample_y = np.append(sample_y, y0)
        
                    # Gather all prediction variations (original, +4π, and -4π)
                    predictions = np.vstack([prediction0, prediction0+ np.pi*2,prediction0 - np.pi*2])
                    # predictions = np.clip(predictions, bound_low, bound_high)
                    len_p = predictions.shape[0]
                    
                    
                                    
                    sample_x = np.concatenate([sample_x, predictions], axis=0)
                    # Extend sample_y with the corresponding y0 values (same for each variation)
                    sample_y = np.concatenate([sample_y, [y0] * len_p])

                    
                    # random points for the next point
                    # if np.abs(y0-optimal[1])< 1e-5 and optimal[1]>1e-3:
                    #     for i in range(2):
                    #         print(f'Cost is too close to optimal: {np.abs(y0-optimal[1])}')
                    #         x0 = np.random.uniform(-np.pi*2,np.pi*2, len(x0)) 
                    #         y0 = func(x0)
                    #         if y0 < optimal[1]:
                    #             optimal = [x0, y0]
                    #         sample_x = np.append(sample_x, [x0], axis=0)
                    #         sample_y = np.append(sample_y, y0)
       
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
            if y < optimal[1]:
                optimal = [prediction, y]
            sample_x = np.append(sample_x, [prediction], axis=0)
            sample_y = np.append(sample_y, y)
        
        res.x = np.copy(optimal[0])
        res.fun = np.copy(optimal[1])
        # res.message = message
        # res.nit = i + 1
        return res
    


    


