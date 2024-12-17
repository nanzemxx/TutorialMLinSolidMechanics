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

class F_to_W_and_P(layers.Layer):
    def __init__(self):
        super(F_to_W_and_P, self).__init__()
        self.ls  = [layers.Dense(16, activation='softplus')]
        self.ls += [layers.Dense(16, activation='softplus', kernel_constraint=constraints.NonNeg())]
        self.ls += [layers.Dense(1,  activation='linear', kernel_constraint=constraints.NonNeg())]
    
    def call(self, inputs):
        F, Cof_F, det_F = inputs
        x_input = tf.concat([F, Cof_F, det_F], axis=-1)  # 保留一个原始输入变量
        with tf.GradientTape() as tape:
            tape.watch(x_input)  # 监视原始输入
            x = x_input
            for layer in self.ls:
                x = layer(x)
            W = x
                
                # 计算 W 对 F 的偏导数 P
        gradients = tape.gradient(W, x_input)
        P = gradients[:, :9]  # 取出 P

        
        return W, P  # 返回 W 和 P
# %%   
"""
main: construction of the NN model

"""

def main():
    # 定义输入：F (9维), Cof(F) (9维), det(F) (1维)
    F_input = tf.keras.Input(shape=[9], name="F_input")
    Cof_F_input = tf.keras.Input(shape=[9], name="Cof_F_input")
    det_F_input = tf.keras.Input(shape=[1], name="det_F_input")
    
    # 自定义层，计算 W 和 P
    W, P = F_to_W_and_P()([F_input, Cof_F_input, det_F_input])
    
    # 构建模型，包含两个输出
    model = tf.keras.Model(inputs=[F_input, Cof_F_input, det_F_input], outputs=[W, P])
    
    # 编译模型，两个输出都使用均方误差损失
    model.compile(optimizer='adam', loss=['mse', 'mse'])
    return model

