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
    path = "D:/vscode/voice/fbank/cre/test"
    tarpath_data = "C:/Users/dhk/Desktop/test_data.txt"
    tarpath_label = "C:/Users/dhk/Desktop/test_label.txt"

    data_set = np.empty((7191, 30000), dtype=float)
    labels = np.empty((7191, ), dtype=int)

    SearchFile(path, ".txt")
    print(len(filelist))
    for i in range(len(filelist)):
        path = filelist[i]
        # print(path)
        cur_id = int(path[-21: -17])
        # print(cur_id)
        labels[i] = cur_id
        temp = np.loadtxt(path, dtype=float)
        temp = temp.T
        # print(temp.shape)
        temp = np.reshape(temp, ((1, -1)))
        data_set[i, :] = temp

    np.savetxt(tarpath_data, data_set)
    np.savetxt(tarpath_label, labels)
    print("ok")


if __name__ == '__main__':
    main()
