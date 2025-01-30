import tensorflow as tf
from tensorflow.keras import layers

# 定义Maxwell模型的RNN单元
class MaxwellRNNCell(tf.keras.layers.Layer):
    def __init__(self, E_inf, E, eta, **kwargs):
        super(MaxwellRNNCell, self).__init__(**kwargs)
        self.E_inf = E_inf  # 参数E∞
        self.E = E  # 参数E
        self.eta = eta  # 参数η
        self.state_size = [[1]]  # 状态大小
        self.output_size = [[1]]  # 输出大小

    def call(self, inputs, states):
        # inputs: 当前步长应变和步长
        eps_n = inputs[0]  # 当前应变
        delta_t = inputs[1]  # 时间步长
        
        # 上一时间步的内部变量γ
        gamma_N = states[0]
        
        # 使用显式欧拉法计算当前的应力和内部变量
        sig_n = self.E_inf * eps_n + self.E * (eps_n - gamma_N)
        gamma_n = gamma_N + delta_t * (1 / self.eta) * (self.E * (eps_n - gamma_N))
        
        return sig_n, [gamma_n]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # 初始化内部变量γ为0
        return [tf.zeros([batch_size, 1], dtype=dtype)]


def main(**kwargs):
    E_inf, E, eta = 0.5, 2, 1
    # 定义输入
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')  # 应变输入
    delta_t = tf.keras.Input(shape=[None, 1], name='input_delta_t')  # 时间步长输入
    
    # 定义Maxwell模型的RNN单元
    cell = MaxwellRNNCell(E_inf, E, eta)
    layer = layers.RNN(cell, return_sequences=True, return_state=False)
    
    # 定义模型
    sigs = layer((eps, delta_t))
    model = tf.keras.Model([eps, delta_t], [sigs])
    return model



