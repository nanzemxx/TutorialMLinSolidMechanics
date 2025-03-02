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
from tensorflow.keras import constraints
import datetime
now = datetime.datetime.now


# %%   
"""
_x_to_y: custom trainable layer

"""
# compare FFNN with ICNN
    
## FFNN
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

#'''
## ICNN
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(16, 'softplus')]
        self.ls += [layers.Dense(16, activation='softplus', kernel_constraint=constraints.NonNeg())]
        
        self.ls += [layers.Dense(1, activation='linear', 
                                 kernel_constraint=constraints.NonNeg())]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x
#''' 
   
'''
## ICNN use other activation functions
class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(16, 'relu')]
        self.ls += [layers.Dense(16, activation='relu', kernel_constraint=constraints.NonNeg())]
        
        self.ls += [layers.Dense(1, activation='linear', 
                                 kernel_constraint=constraints.NonNeg())]
            
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