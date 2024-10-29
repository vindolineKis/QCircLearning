import tensorflow as tf

import keras
from keras.src.models.sequential import Sequential
from keras import layers as klayers
from keras import optimizers as koptimizers
import numpy as np
from scipy.optimize import minimize
# from tensorflow.keras.optimizers import Adam
from scipy.optimize import approx_fprime
from scipy.misc import derivative

from typing import List, Callable
from keras import backend as K


class trainer_model(Sequential):
    def __init__(self,
                 layers:klayers.Layer=None,
                 trainable:bool=True,
                 name:str=None,
                 ):
        super().__init__(layers=layers, trainable=trainable, name=name)
    K.set_floatx('float64') 
    # TODO: gradient based methods
   
    def back_minimize(self,
                 x0:np.ndarray=None,
                 method = 'BFGS', verbose = 0):
        """
        After the model is trained, minimize the output by training the input.
        """

        # # @tf.function
        def to_minimize(x):
            # pad_x = np.array([x])
            return self(x)

        if x0 is None:
            x = np.random.rand(self.inputs[0].shape[1])
        else:
            x = x0
        def to_minimize_with_grad(x):
            x0 = x
            x = tf.Variable([x0])
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = to_minimize(x)
            gradients = tape.gradient(loss, x)
            # print("shape of the gradients:", gradients.numpy().shape())

            if gradients is None:
                raise ValueError("Gradient calculation failed; TensorFlow returned None.")

            gradients = gradients.numpy()[0]
            
            # for i in range(len(x0)):
            #     if x0[i] > np.pi/2:
            #         gradients[i]= abs(x0[i])*20
            #         # overwritten_count += 1
            #     elif x0[i] < -np.pi/2:
            #         gradients[i]= -abs(x0[i])*20
            #         # overwritten_count += 1
            #     else:
            #         gradients[i] = gradients[i]
        
            # print x0, gradients
            # print(f'parameters: {x0}')
            # print(f"Using gradients: {gradients}")
        

            return loss.numpy()[0][0], gradients
        
        def diff_grad(x, f, h):
            delta = np.zeros_like(x)
            delta[0] = h

            grad = []

            for i in range(len(x)):
                d = np.roll(delta, i)
                grad.append((f(x + d) - f(x - d)) / (2 * h))

            return np.array(grad)
        
        def finite_difference_grad(x, h=1e-6):
            # print("self(x)is:",self(x))
            f = lambda x: to_minimize(np.atleast_2d(x)).numpy()[0][0]
            # f = to_minimize(x).numpy()[0][0]
            # f = lambda x: to_minimize(x)
            # f = self([x])
            gradients = approx_fprime(x, f, h)
            gradients_m = diff_grad(x, f, h)
          
            # gradients_m = gradients.numpy
            loss = f(x)
            return loss, gradients,gradients_m
        
        def compare_gradients(x):
            _, tf_grad = to_minimize_with_grad(x)
            _, fd_grad ,fd_grad_m= finite_difference_grad(x, h=1e-6)
            
           
            
            print("TensorFlow gradient:", tf_grad)
            # print(tf_grad.shape())
            print("Finite difference gradient:", fd_grad)
            print("Finite difference gradient_m:", fd_grad_m)
            # print(fd_grad.shape())
            
            grad_diff = np.linalg.norm(tf_grad - fd_grad)
            print(f"Gradient difference (L2 norm): {grad_diff}")
            return grad_diff
        
        print('the difference between the gradients of the finite difference and the tensorflow is:', compare_gradients(x))

        result = minimize(to_minimize_with_grad, x, bounds=[(-np.pi/2,np.pi/2)]*len(x0), jac=True, method=method,tol=1e-7)
        
        if result.success:
            print("Optimization converged successfully.")
        else:
             print("Optimization did not converge. Reason:", result.message)

        # result = minimize(to_minimize_with_grad, x, jac=True, method=method,tol=1e-6)

        # if np.linalg.norm(result.x - x0) > 1e-3:
        #     print(f'Prediction is different from in trainner: {np.linalg.norm(result.x - x0)} after {result.nit} iterations')

        return result.x 


    @staticmethod
    def default_model(input_shape:tuple):
        initializer = keras.initializers.he_normal(seed=10)
        return trainer_model(
            layers=[
                klayers.Input(input_shape),
                klayers.Dense(96, activation='elu',
                                   kernel_initializer=initializer,
                                   kernel_regularizer=keras.regularizers.l2(1e-8),
                                   ),
                klayers.Dense(64, activation='elu'),
                klayers.Dense(18, activation='elu'),
                klayers.Dense(10, activation='elu'),
                klayers.Dense(1),
                ],
                name='default_model'
            )

    @staticmethod
    def simple_model(input_shape:tuple):
        return trainer_model(
            layers=[
                klayers.Input(input_shape),
                klayers.Dense(32, activation='elu'),
                klayers.Dense(8, activation='sigmoid'),
                klayers.Dense(1),
                ],
                name='simple_model'
            )