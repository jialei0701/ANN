import numpy as np
import os
import sys

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
    # path = "D:/vscode/voice/fbank/cre/test"
    path = "D:/test"
    writer = tf.python_io.TFRecordWriter('test.tfrecord')

    SearchFile(path, ".txt")
    print(len(filelist))
    for i in range(2):
        path = filelist[i]
        # print(path)
        cur_id = int(path[-21: -17]) - 1081
        print(cur_id)

        temp = np.loadtxt(path, dtype=float)
        temp = temp.T
        # print(temp.shape)
        temp = np.reshape(temp, (120*250, ))
        # print(temp.shape)

    print("ok")


if __name__ == '__main__':
    main()
