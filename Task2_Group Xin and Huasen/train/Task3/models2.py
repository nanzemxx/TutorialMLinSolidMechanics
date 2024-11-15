import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import constraints

class _C_F_to_W(layers.Layer):
    def __init__(self):
        super(_C_F_to_W, self).__init__()
        self.dense1 = layers.Dense(4, activation='softplus')
        self.dense2 = layers.Dense(4, activation='softplus', kernel_constraint=tf.keras.constraints.NonNeg(), bias_constraint=tf.keras.constraints.NonNeg())
        self.dense3 = layers.Dense(1, activation='linear', kernel_constraint=tf.keras.constraints.NonNeg())
    
    def call(self, C, F):
        # 使用已经定义好的层
        x = tf.concat([C, F], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)
    
class _W_to_P(layers.Layer):
    def __init__(self):
        super(_W_to_P, self).__init__()
        self.C_F_to_W = _C_F_to_W()
        
    def call(self, C, F):
        # 使用 tf.GradientTape 来计算 W 对 F 的偏导数
        with tf.GradientTape() as tape:
            tape.watch(F)  # 确保 F 被监听
            W = self.C_F_to_W(C, F)  # 计算 W
        P = tape.gradient(W, F)  # 计算偏导数 P
        return P


def main(**kwargs):
    # Define inputs
    Cs = tf.keras.Input(shape=[5])  # Input C
    Fs = tf.keras.Input(shape=[9])  # Input F
    
    # Get W using the custom _C_to_W layer
    Ws = _C_F_to_W()(Cs, Fs)
    
    # Compute P = ∂W/∂F using the custom _W_to_P layer
    Ps = _W_to_P()(Cs, Fs)
    
    # Define the model
    model = tf.keras.Model(inputs=[Cs, Fs], outputs=[Ws, Ps])
    
    # Compile the model with optimizer and loss function
    model.compile(optimizer='adam', loss='mse')
    
    return model
