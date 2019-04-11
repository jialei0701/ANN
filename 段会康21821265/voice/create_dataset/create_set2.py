import numpy as np
import os
import tensorflow as tf

filelist = []


def SearchFile(path, str_suffix):
    try:
        files = os.listdir(path)

        for f in files:
            fl = os.path.join(path, f)
            if os.path.isdir(fl):
                SearchFile(fl, str_suffix)
            elif os.path.isfile(fl) and f.endswith(str_suffix):
                # print(type(fl))
                fl = fl.replace('\\', '/')
                # print(fl)
                filelist.append(fl)

    except Exception:
        print(u'+++++++++++')


def main():
    path = "D:/vscode/voice/fbank/cre/train"
    tarpath = "D:/vscode/voice/fbank/tfrecord/train_600.tfrecord"
    writer = tf.python_io.TFRecordWriter(tarpath)

    SearchFile(path, ".txt")
    print(len(filelist))

    pre_id = -1
    for i in range(len(filelist)):
        path = filelist[i]
        # print(path)
        cur_id = int(path[-21:-17]) - 1081
        # print(cur_id)
        temp = np.loadtxt(path, dtype=float)
        # print(temp.shape)
        assert temp.shape == (250, 120), "shape is not [250, 120]"
        assert temp.shape[0] == 250, "shape_0 is not 250"
        assert temp.shape[1] == 120, "shape_1 is not 120"

        data = np.reshape(temp, (250 * 120, ))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'data':
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=data)),
                    'label':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[cur_id]))
                }))

        writer.write(example.SerializeToString())

        if pre_id != cur_id:
            print(cur_id, "ok")
            pre_id = cur_id
    writer.close()


if __name__ == '__main__':
    main()
