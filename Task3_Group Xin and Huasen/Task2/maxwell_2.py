import tensorflow as tf
from tensorflow.keras import layers


class MaxwellRNNCellWithNN(tf.keras.layers.Layer):
    def __init__(self, E_inf, E, hidden_units=[16, 16, 16], **kwargs):
      
        super(MaxwellRNNCellWithNN, self).__init__(**kwargs)
        self.E_inf = E_inf
        self.E = E
        
     
        self.ffnn = tf.keras.Sequential()
        for units in hidden_units:
            self.ffnn.add(layers.Dense(units, activation='softplus'))
        
        self.ffnn.add(layers.Dense(1, activation=None))
        
       
        self.state_size = [1]  
        self.output_size = [1] 

    def call(self, inputs, states):
      
        eps_n = inputs[0]
        delta_t = inputs[1]
        
        gamma_N = states[0]
        
    
        nn_input = tf.concat([eps_n, gamma_N], axis=-1)
        gamma_dot = self.ffnn(nn_input) 
    
        gamma_n = gamma_N + delta_t * gamma_dot

        sig_n = self.E_inf * eps_n + self.E * (eps_n - gamma_n)
        

        return sig_n, [gamma_n]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
     
        return [tf.zeros([batch_size, 1], dtype=dtype)]


def main(E_inf=0.5, E=2.0, hidden_units=[16,16,16]):
   
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')
    delta_t = tf.keras.Input(shape=[None, 1], name='input_delta_t')
    
    
    cell = MaxwellRNNCellWithNN(E_inf, E, hidden_units)
    
    
    layer = layers.RNN(cell, return_sequences=True, return_state=False)
    
   
    sigs = layer((eps, delta_t))
    
   
    model = tf.keras.Model(inputs=[eps, delta_t], outputs=sigs)
    model.compile('adam', 'mse')
    return model

