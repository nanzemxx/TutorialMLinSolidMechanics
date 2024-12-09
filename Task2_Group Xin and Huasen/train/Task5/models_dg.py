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

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.out_layer = layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out_layer(x)

    def train_step(self, data):
        # data是fit传入的数据，通常为 (X_batch, (W_batch, P_batch))
        X_batch, y_batch = data
        W_batch, P_batch = y_batch  # y_batch可以是tuple或list，此处假设传入(W_batch, P_batch)

        # 前9维为F，需要对其求导。要确保X_batch是张量且可求导。
        # 通常X_batch是tf.Tensor但非Variable，我们可以在tape中watch它。
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape_F:
                tape_F.watch(X_batch)
                W_pred = self(X_batch, training=True)
            
            # 对W关于F求导
            # 假设X_batch[:, :9]为F，求导后形状为 (batch_size, 9)
            dW_dF = tape_F.gradient(W_pred, X_batch)[:, :9]

            # 计算损失
            loss_W = tf.reduce_mean((W_pred - W_batch)**2)
            loss_P = tf.reduce_mean((dW_dF - P_batch)**2)
            loss = loss_W + loss_P

        # 对模型参数求导并更新
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # 这里返回的字典会显示在fit的日志中
        return {"loss": loss, "loss_W": loss_W, "loss_P": loss_P}

# %%   
"""
main: construction of the NN model

"""

def main(**kwargs):
    model = MyModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    return model

