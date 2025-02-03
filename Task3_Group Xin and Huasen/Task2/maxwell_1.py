import tensorflow as tf
from tensorflow.keras import layers


class MaxwellRNNCell(tf.keras.layers.Layer):
    def __init__(self, E_inf, E, eta, **kwargs):
        super(MaxwellRNNCell, self).__init__(**kwargs)
        self.E_inf = E_inf  
        self.E = E  
        self.eta = eta  
        self.state_size = [[1]]  
        self.output_size = [[1]]  

    def call(self, inputs, states):
        
        eps_n = inputs[0]  
        delta_t = inputs[1]  
        
      
        gamma_N = states[0]
        
     
        sig_n = self.E_inf * eps_n + self.E * (eps_n - gamma_N)
        gamma_n = gamma_N + delta_t * (1 / self.eta) * (self.E * (eps_n - gamma_N))
        
        return sig_n, [gamma_n]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
     
        return [tf.zeros([batch_size, 1], dtype=dtype)]


def main(**kwargs):
    E_inf, E, eta = 0.5, 2, 1
 
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')  
    delta_t = tf.keras.Input(shape=[None, 1], name='input_delta_t')  
    
 
    cell = MaxwellRNNCell(E_inf, E, eta)
    layer = layers.RNN(cell, return_sequences=True, return_state=False)
    

    sigs = layer((eps, delta_t))
    model = tf.keras.Model([eps, delta_t], [sigs])
    return model



