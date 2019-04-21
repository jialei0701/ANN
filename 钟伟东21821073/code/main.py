#coding=utf-8
import numpy as np
import tensorflow as tf
import time

turns = 20
T, N, d = 1., 1, 64  # d为方程x的维数
batch_size = 8192
neurons = [d + 100, d + 100, 1]

model = tf.keras.models.Sequential([
  tf.keras.layers.BatchNormalization(input_shape=[d]),
  tf.keras.layers.Dense(neurons[0], use_bias=False),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Activation('tanh'),
  tf.keras.layers.Dense(neurons[1], use_bias=False),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Activation('tanh'),
  tf.keras.layers.Dense(neurons[2], use_bias=False),
  tf.keras.layers.BatchNormalization()
])



model.compile(optimizer='adam', loss='mse')




# data = []
# for i in np.arange(0, 1, 0.05):
#     for j in np.arange(0, 1, 0.05):
#         data.append([i,j] + list(np.random.random(62)))
# data = np.array(data)

def phi(x):
    return np.sum(x**2, axis=1, keepdims=True)


file_name = "./data.csv"
file_out = open(file_name, 'w')
logfile_out = open("./log.txt", 'w')
# file_out.write('')


# mc_rounds, mc_freq = 10, 10
mc_rounds, mc_freq = 500, 50

for e in range(1,turns):
    # print(e)

    xi = np.random.uniform(0,1,[batch_size, d])
    # xi = np.random.random([batch_size, d])
    x_sde = xi + np.random.normal(0, np.sqrt(2. * T / N), [batch_size, d])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f'model/pde-{e}.hdf5')
    ]
    model.fit(xi, phi(x_sde), verbose=0, epochs=mc_freq, callbacks=callbacks)

    # if e % mc_freq==0:
    # mc loss
    # print('mc start!!!!!!!!!!!!!!!!')
    l1_err, l2_err, li_err = 0., 0., 0.
    rel_l1_err, rel_l2_err, rel_li_err = 0., 0., 0.
    for mc_e in range(mc_rounds):
        t1 = time.time()
        xi = np.random.uniform(0,1,[batch_size, d])
        u_reference = phi(xi) + 2. * T * d
        t2 = time.time()
        u_approx = model.predict(xi)

        t3 = time.time()
        err = np.abs(u_approx - u_reference)
        l1_err += np.mean(err)
        # print(u_approx)
        # print(u_reference)
        l2_err += np.mean(err**2)
        li_err = np.maximum(li_err, np.max(err))
        
        rel_err = err / np.max(u_reference)
        rel_l1_err += np.mean(rel_err)
        rel_l2_err += np.mean(rel_err ** 2)
        rel_li_err = np.maximum(rel_li_err, np.max(rel_err))
        t4 = time.time()
        # print(t2-t1, t3-t2, t4-t3)
    l1_err, l2_err = l1_err / mc_rounds, np.sqrt(l2_err /mc_rounds)
    rel_l1_err, rel_l2_err = rel_l1_err / mc_rounds, np.sqrt(rel_l2_err / mc_rounds)
    print('%i, %f, %f, %f, %f, %f, %f\n' % (e, l1_err, l2_err, li_err, rel_l1_err, rel_l2_err, rel_li_err))
    logfile_out.write('%i, %f, %f, %f, %f, %f, %f\n' % (e, l1_err, l2_err, li_err, rel_l1_err, rel_l2_err, rel_li_err))
    logfile_out.flush()
    # data结果可视化
    u_test = model.predict(data)
    u_test_reference = phi(data) + 2. * T * d
    # print(u_test.shape)
    # print(u_test)
    u_test = [i[0] for i in u_test]
    file_out.write(str(u_test)[1:-1])
    file_out.write('\n')
    # file_out.write('%i, %f, %f, %f, %f, %f, %f, %f, %f, %f\n' % (gs, l1_err, l2_err, li_err, rel_l1_err, rel_l2_err, rel_li_err, lr, t1_train-t0_train, t_mc-t1_train))
    file_out.flush()