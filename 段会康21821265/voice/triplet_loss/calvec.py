# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf


def create_embedding(data_path, label_path, out_path):
    vec = np.loadtxt(data_path)
    label = np.loadtxt(label_path)

    # print(label.shape)
    # 当label中只有一个label时会报错，label.shape为（）
    len0 = label.shape[0]

    cnt1 = 0
    list_num = []
    preid = label[0]
    for i in range(len0):
        curid = label[i]
        if curid == preid:
            cnt1 += 1
            if i == (len0 - 1):
                list_num.append(cnt1)
                cnt1 = 0
        else:
            list_num.append(cnt1)
            cnt1 = 1
        preid = curid
    if cnt1 != 0:
        list_num.append(cnt1)
    # print(list_num)

    num2 = len(list_num)
    # print(type(num2))
    # print(num2)
    final = np.zeros((num2, 128), dtype=float)
    p = 0
    for i in range(num2):
        temp = np.zeros((128, ), dtype=float)
        for j in range(list_num[i]):
            temp += vec[p]
            p += 1
        final[i] = temp / list_num[i]

    res = tf.nn.l2_normalize(final, dim=1)

    with tf.Session() as sess:
        res = sess.run(res)

        # sum = 0
        # for i in range(128):
        #     sum += res[0][i] * res[0][i]
        # print(sum)

        np.savetxt(out_path, res)


def fun1():
    data_path1 = "embeddings/model.txt"
    label_path1 = "embeddings/test_label.txt"
    out_path1 = "embeddings/model_final.txt"

    data_path2 = "embeddings/vec.txt"
    label_path2 = "embeddings/test_400_labels.txt"
    out_path2 = "embeddings/vec_final.txt"

    create_embedding(data_path1, label_path1, out_path1)
    create_embedding(data_path2, label_path2, out_path2)


if __name__ == "__main__":
    fun1()