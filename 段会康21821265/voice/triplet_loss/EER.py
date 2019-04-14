import numpy as np
import matplotlib.pyplot as plt


def cal_dist():
    model = np.loadtxt("embeddings/model_final.txt")
    vec = np.loadtxt("embeddings/vec_final.txt")
    num_model = model.shape[0]
    num_vec = vec.shape[0]

    # print(model.shape[0])
    # print(vec.shape[0])

    mat_dis = np.empty((100, 400), dtype=float)
    for i in range(num_model):
        for j in range(num_vec):
            v1 = model[i]
            v2 = vec[j]
            d = np.sqrt(np.sum(np.square(v1 - v2)))
            mat_dis[i][j] = d

    np.savetxt("dis.txt", mat_dis)

    return mat_dis


def draw_roc(mat1):
    eer = 0
    best_vth = 0
    FAR = []
    FRR = []
    # print(np.max(mat1))
    # print(np.min(mat1))
    th = list(range(120, 1600, 1))
    for num in th:
        d_th = num * 1.0 / 1000
        num_false = 0
        num_refuse = 0
        for i in range(mat1.shape[0]):
            for j in range(mat1.shape[1]):
                st = 4 * i
                ed = 4 * (i + 1)
                if j >= st and j < ed:
                    if mat1[i][j] > d_th:
                        num_refuse += 1
                else:
                    if mat1[i][j] <= d_th:
                        num_false += 1

        num1 = (num_false * 1.0) / 396
        num2 = (num_refuse * 1.0) / 4
        FAR.append(num1)
        FRR.append(num2)
        if num1 >= (num2 - 0.2) and num1 <= (num2 + 0.2):
            eer = num1
            best_vth = d_th

    # print(FAR)
    # print("----")
    # print(FRR)
    # print(eer)
    # print(best_vth)
    # plt.figure()
    # plt.plot(FAR, FRR)
    # plt.show()
    return eer,best_vth


def find_eer():
    mat_dis = cal_dist()
    eer, best_vth = draw_roc(mat_dis)

    return eer, best_vth


if __name__ == "__main__":
    find_eer()