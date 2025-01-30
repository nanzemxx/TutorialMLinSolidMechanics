import tensorflow as tf
from tensorflow.keras import layers

# 定义Maxwell模型的RNN单元（使用FFNN表示演化方程）
class MaxwellRNNCellWithFFNN(tf.keras.layers.Layer):
    def __init__(self, E_inf, E, **kwargs):
        super(MaxwellRNNCellWithFFNN, self).__init__(**kwargs)
        self.E_inf = E_inf  # 参数E∞
        self.E = E  # 参数E
        self.state_size = [[1]]  # 状态大小
        self.output_size = [[1]]  # 输出大小
        
        # 定义前馈神经网络（FFNN）
        self.ffnn = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # 输出为更新量
        ])

    def call(self, inputs, states):
        # inputs: 当前步长应变和步长
        eps_n = inputs[0]  # 当前应变
        delta_t = inputs[1]  # 时间步长
        
        # 上一时间步的内部变量γ
        gamma_N = states[0]
        
        # 计算应力
        sig_n = self.E_inf * eps_n + self.E * (eps_n - gamma_N)
        
        # 使用FFNN更新内部变量γ
        ffnn_input = tf.concat([eps_n, delta_t, gamma_N], axis=1)
        gamma_update = self.ffnn(ffnn_input)
        gamma_n = gamma_N + gamma_update  # 更新内部变量
        
        return sig_n, [gamma_n]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # 初始化内部变量γ为0
        return [tf.zeros([batch_size, 1], dtype=dtype)]


def main(**kwargs):
    # 定义输入
    E_inf, E = 0.5, 2
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')  # 应变输入
    delta_t = tf.keras.Input(shape=[None, 1], name='input_delta_t')  # 时间步长输入
    
    # 定义Maxwell模型的RNN单元（带FFNN）
    cell = MaxwellRNNCellWithFFNN(E_inf, E)
    layer = layers.RNN(cell, return_sequences=True, return_state=False)
    
    # 定义模型
    sigs = layer((eps, delta_t))
    model = tf.keras.Model([eps, delta_t], [sigs])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model



