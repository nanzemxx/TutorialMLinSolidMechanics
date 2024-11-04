"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein
         
08/2022
"""


# %%   
"""
Import modules

"""
import tensorflow as tf
from tensorflow.keras import layers
import datetime
now = datetime.datetime.now


# %%   
"""
_x_to_y: custom trainable layer

"""
# Vary the number of hidden layers
## 1 hidden layer
''' 
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(16, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x
'''
## 2 hidden layer
''' 
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(16, 'softplus')]
        self.ls += [layers.Dense(16, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x
'''
## 3 hidden layer
#'''
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(16, 'softplus')]
        self.ls += [layers.Dense(16, 'softplus')]
        self.ls += [layers.Dense(16, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x
#'''

# Vary the number of nodes
## 4 nodes
'''
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(4, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x
'''
## 8 nodes
'''
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(8, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x
'''
## 16 nodes
'''
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(16, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x
'''

# Vary the number of epochs
''' 
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x
'''
# Use different activation functions
## Relu
''' 
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(16, 'Relu')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x
'''
## Softplus
''' 
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(16, 'Softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x
'''
## Sigmoid
''' 
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(16, 'Sigmoid')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x
'''
# %%   
"""
main: construction of the NN model

"""

def main(**kwargs):
    # define input shape
    xs = tf.keras.Input(shape=[1])
    # define which (custom) layers the model uses
    ys = _x_to_y(**kwargs)(xs)
    # connect input and output
    model = tf.keras.Model(inputs = [xs], outputs = [ys])
    # define optimizer and loss function
    model.compile('adam', 'mse')
    return model