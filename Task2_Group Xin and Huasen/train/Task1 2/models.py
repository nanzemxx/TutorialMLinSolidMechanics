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

class _C_to_P(layers.Layer):
    def __init__(self):
        super(_C_to_P, self).__init__()
        self.ls = [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(9, activation='linear')]
    
    def call(self, C):
        for l in self.ls:
            C = l(C)
        return C
# %%   
"""
main: construction of the NN model

"""

def main(**kwargs):
    # define input shape
    Cs = tf.keras.Input(shape=[6])
    # define which (custom) layers the model uses
    Ps = _C_to_P(**kwargs)(Cs)
    # connect input and output
    model = tf.keras.Model(inputs = [Cs], outputs = [Ps])
    # define optimizer and loss function
    model.compile('adam', 'mse')
    return model

