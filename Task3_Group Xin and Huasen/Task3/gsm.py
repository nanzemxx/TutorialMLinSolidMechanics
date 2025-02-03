import tensorflow as tf
from tensorflow.keras import layers

class GSMRNNCellWithFFNN(tf.keras.layers.Layer):
    def __init__(self, g, **kwargs):
        super(GSMRNNCellWithFFNN, self).__init__(**kwargs)
        self.g = g  # g = η^(-1)
        self.state_size = [[1]]
        self.output_size = [[1]]
        
        
        self.energy_ffnn = tf.keras.Sequential([
            layers.Dense(32, activation='softplus'),
            layers.Dense(16, activation='softplus'),
            layers.Dense(16, activation='softplus'),
            layers.Dense(1, name='energy')  
        ])
        
    def energy_function(self, eps, gamma):
        
        inputs = tf.concat([eps, gamma], axis=1)
        return self.energy_ffnn(inputs)
    
    def compute_gradients(self, eps, gamma):

        with tf.GradientTape(persistent=True) as tape:
           
            tape.watch(eps)
            tape.watch(gamma)
            
            energy = self.energy_function(eps, gamma)
            
        
        de_deps = tape.gradient(energy, eps)
        
        de_dgamma = tape.gradient(energy, gamma)
        
        de_deps = tf.clip_by_value(de_deps, clip_value_min=-1e3, clip_value_max=1e3)
        de_dgamma = tf.clip_by_value(de_dgamma, clip_value_min=-1e3, clip_value_max=1e3)
        
        del tape  
        return de_deps, de_dgamma

    def call(self, inputs, states):
        
        eps_n = inputs[0]  
        delta_t = inputs[1]  
        gamma_n = states[0]  
        
        
        eps_n = tf.convert_to_tensor(eps_n)
        gamma_n = tf.convert_to_tensor(gamma_n)
        
        
        de_deps, de_dgamma = self.compute_gradients(eps_n, gamma_n)
        
        
       
        dgamma_dt = -self.g * de_dgamma
        
        
        gamma_next = gamma_n + dgamma_dt * delta_t
        
       
        sigma = de_deps
        
        return sigma, [gamma_next]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([batch_size, 1], dtype=dtype)]


def main(**kwargs):
    
    g = 1.0  
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')
    delta_t = tf.keras.Input(shape=[None, 1], name='input_delta_t')
    
    
    cell = GSMRNNCellWithFFNN(g)
    layer = layers.RNN(cell, return_sequences=True, return_state=False)
    
    
    sigs = layer((eps, delta_t))
    model = tf.keras.Model([eps, delta_t], [sigs])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model