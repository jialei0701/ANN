# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


def _parse_record(record):
    features = {
        'data': tf.FixedLenFeature([250 * 120], dtype=tf.float32),
        'label': tf.FixedLenFeature([], dtype=tf.int64)
    }
    parsed_features = tf.parse_single_example(record, features=features)

    data = parsed_features['data']
    label = parsed_features['label']
    return data, label


def read_test(input_file):

    test_size = 7191
    # 用 dataset 读取 tfrecord 文件
    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(_parse_record)
    print(dataset.output_types)
    print(dataset.output_shapes)
    train_iterator = dataset.make_one_shot_iterator()
    ele = train_iterator.get_next()

    labels = np.empty((test_size, ), dtype=int)
    with tf.Session() as sess:
        for i in range(test_size):
            data, label = sess.run(ele)
            # print(data)
            # print(label)
            # print(features)
            labels[i] = label
    np.savetxt("test_label.txt", labels)


# read_test("txt2.tfrecord")
if __name__ == '__main__':
    read_test("D:/vscode/voice/fbank/tfrecord/test.tfrecord")
