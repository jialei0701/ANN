import numpy as np
import tensorflow as tf
from common import kolmogorov_train_and_test

tf.reset_default_graph()
dtype = tf.float32
T, N, d = 1., 1, 100  # d为方程x的维数
batch_size = 8192
neurons = [d + 100, d + 100, 1]

train_steps = 750000

mc_rounds, mc_freq = 1250, 100

# 学习率变化
lr_boundaries = [250001, 500001]
lr_values = [0.001, 0.0001, 0.00001]
# 均匀
xi = tf.random_uniform(shape=(batch_size, d), minval=0., maxval=1., dtype=dtype)  # xi为均匀采样，在d维空间的[a, b]区间上
# 正态
x_sde = xi + tf.random_normal(shape=(batch_size, d), stddev=np.sqrt(2. * T / N), dtype=dtype)


# with tf.Session() as sess:
#     sess.run(xi)

def phi(x):
    return tf.reduce_sum(x**2, axis=1, keepdims=True)


u_reference = phi(xi) + 2. * T * d
kolmogorov_train_and_test(xi, x_sde, phi, u_reference, neurons, lr_boundaries, lr_values, train_steps, mc_rounds, mc_freq, 'example3_2.csv', dtype)
