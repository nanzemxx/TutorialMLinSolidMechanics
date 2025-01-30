import tensorflow as tf
from tensorflow.keras import layers

class GSMRNNCellWithFFNN(tf.keras.layers.Layer):
    def __init__(self, g, **kwargs):
        super(GSMRNNCellWithFFNN, self).__init__(**kwargs)
        self.g = g  # g = η^(-1)
        self.state_size = [[1]]
        self.output_size = [[1]]
        
        # FFNN to represent the energy function e(ε,γ)
        self.energy_ffnn = tf.keras.Sequential([
            layers.Dense(16, activation='softplus'),
            layers.Dense(16, activation='softplus'),
            layers.Dense(16, activation='softplus'),
            layers.Dense(1, name='energy')  # 输出为能量值
        ])
        
    def energy_function(self, eps, gamma):
        # 将应变和内部变量连接作为能量函数的输入
        inputs = tf.concat([eps, gamma], axis=1)
        return self.energy_ffnn(inputs)
    
    def compute_gradients(self, eps, gamma):
        """
        计算能量函数对应变和内部变量的梯度
        """
        with tf.GradientTape(persistent=True) as tape:
            # 需要追踪这些变量的梯度
            tape.watch(eps)
            tape.watch(gamma)
            # 计算能量
            energy = self.energy_function(eps, gamma)
            
        # 计算对应变的梯度 (∂e/∂ε)
        de_deps = tape.gradient(energy, eps)
        # 计算对内部变量的梯度 (∂e/∂γ)
        de_dgamma = tape.gradient(energy, gamma)
        
        de_deps = tf.clip_by_value(de_deps, clip_value_min=-1e3, clip_value_max=1e3)
        de_dgamma = tf.clip_by_value(de_dgamma, clip_value_min=-1e3, clip_value_max=1e3)
        
        del tape  # 释放资源
        return de_deps, de_dgamma

    def call(self, inputs, states):
        # 获取输入
        eps_n = inputs[0]  # 当前应变
        delta_t = inputs[1]  # 时间步长
        gamma_n = states[0]  # 当前内部变量
        
        # 将张量转换为可追踪的变量
        eps_n = tf.convert_to_tensor(eps_n)
        gamma_n = tf.convert_to_tensor(gamma_n)
        
        # 计算能量函数的梯度
        de_deps, de_dgamma = self.compute_gradients(eps_n, gamma_n)
        
        # GSM模型的演化方程
        # dγ/dt = -g * ∂e/∂γ
        dgamma_dt = -self.g * de_dgamma
        
        # 更新内部变量
        gamma_next = gamma_n + dgamma_dt * delta_t
        
        # 应力等于能量对应变的梯度
        sigma = de_deps
        
        return sigma, [gamma_next]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([batch_size, 1], dtype=dtype)]


def main(**kwargs):
    # 定义输入
    g = 1.0  # 示例值，可以根据需要调整
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')
    delta_t = tf.keras.Input(shape=[None, 1], name='input_delta_t')
    
    # 创建GSM模型
    cell = GSMRNNCellWithFFNN(g)
    layer = layers.RNN(cell, return_sequences=True, return_state=False)
    
    # 构建模型
    sigs = layer((eps, delta_t))
    model = tf.keras.Model([eps, delta_t], [sigs])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model