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
import numpy as np


# %%   
"""
Generate data for a bathtub function

"""

def bathtub():
    
    x_values = np.linspace(-4, 4, 20)
    y_values = np.linspace(-4, 4, 20)
    
    
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    ys_values = x_grid**2 - y_grid**2  
    
    
    xs = np.vstack((x_grid.flatten(), y_grid.flatten())).T  
    ys = ys_values.flatten()  

    
    
    xs_c = np.vstack((xs[:240, :], xs[330:400, :]))  
    ys_c = np.concatenate([ys[:240], ys[330:400]])
    
    xs = tf.convert_to_tensor(xs, dtype=tf.float32)   
    ys = tf.expand_dims(ys, axis=1) 
    
    xs_c = tf.convert_to_tensor(xs_c, dtype=tf.float32)  
    ys_c = tf.expand_dims(ys_c, axis=1)  
    
    return xs, ys, xs_c, ys_c, x_grid, y_grid
