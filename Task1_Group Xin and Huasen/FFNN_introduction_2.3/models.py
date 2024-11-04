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
from tensorflow.keras import constraints
now = datetime.datetime.now


# %%   
"""
_x_to_y: custom trainable layer

"""

class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        
        self.ls = [layers.Dense(16, 'softplus')]
        self.ls += [layers.Dense(16, activation='softplus', kernel_constraint=constraints.NonNeg())]
        
        self.ls += [layers.Dense(1, activation='linear', 
                                 kernel_constraint=constraints.NonNeg())]

    def call(self, x):
        x_input = x  
        with tf.GradientTape() as tape:
            tape.watch(x_input)
            for l in self.ls:
                x = l(x)
            output = x
        gradient = tape.gradient(output, x_input)  
        return output, gradient
# %%   
"""
main: construction of the NN model

"""

def main(**kwargs):
    
    xs = tf.keras.Input(shape=[2])
   
    ys, grads = _x_to_y(**kwargs)(xs)
    
    model = tf.keras.Model(inputs = [xs], outputs=[ys, grads])
    #Calibration on f2
    #model.compile(optimizer='adam', loss='mse', loss_weights=[1.0, 0.0])
    #Calibration on f2 and its gradient
    #model.compile(optimizer='adam', loss='mse', loss_weights=[1.0, 1.0])
    #Calibration on gradient
    model.compile(optimizer='adam', loss='mse', loss_weights=[0.0, 1.0])
    return model

