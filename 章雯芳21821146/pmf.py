"""
Implementation of Probabilistic Matrix Factorization
The component of baseline EMCDR framework
通过测试，参数可以如下定义：
eta = 0.1
lambda_u = lambda_v = 0.1
"""
import numpy as np
import matplotlib.pyplot as plt


def trainPMF(train, K, eta, lambda_u, lambda_v, maxEpoch, batch_size=1):
    """
    This is the function of Probabilistic Matrix Factorization
    :param train: the rating matrix for training set, a numpy array, size is n * m
    :param K: 隐向量的维数
    :param eta: learning rate
    :param lambda_u: λ(u)
    :param lambda_v: λ(v)
    :param maxEpoch: the max iteration step
    :param batch_size: the number of user in each batch, default by 1 user per batch
    :return: 返回U 和 V, 增加返回loss
    目标函数(最小化):
    E = 1/2 * (sum(sum(I[i][j] * (R[i][j] - U[i]V[j])^2)) + λ(u)/2 * sum(U[i]^2) + λ(v)/2 * sum(V[i]^2)
    """

    # 获得rating的纬度
    n, m = train.shape

    # 获得指示矩阵 indicator
    idct = np.where(train > 0, 1, 0)

    # 映射原来的评分矩阵到[0, 1]
    #train = t(train)

    # 随机生成用户特征矩阵和物品特征矩阵
    # 使用标准正太分布
    U = 0.1 * np.random.normal(0, 0.01, (n, K))
    V = 0.1 * np.random.normal(0, 0.01, (m, K))

    # batch number
    batch_num = n // batch_size
    if n % batch_size != 0:
        batch_num += 1
    # gradient decrease with mini-batch
    losses = []  # the loss of each epoch
    rmses = []

    for ii in range(maxEpoch):
        loss = 0.0  # 目标函数
        batch = 0  # the number of ratings that have been visited
        # 加入mini-batch的算法流程, 每个batch是按照用户数量来计量的
        for batch in range(batch_num):  # loop batches
            item_gr = np.zeros((m, K), dtype='float64')  # gradient for items
            # each user per batch
            for i in range(batch_size):
                u = batch_size * batch + i  # the user who is visited next
                if u >= n:  # 遍历了所有用户 break
                    break
                user_gr = np.zeros(K)  # the sum of gradients of this user

                # loop items
                for j in range(m):
                    if idct[u][j] != 0:  # there is a rating for user u to item j
                        # pr = np.sum(np.multiply(U[i], V[j]))    # pr = prediction rating
                        pr = np.dot(U[u], V[j])
                        delta = train[u][j] - pr
                        try:
                            user_gr += delta * V[j] - lambda_u * U[u]
                            item_gr[j] += delta * U[u] - lambda_v * V[j]  # item j gr sum
                        except Exception as e:
                            print(e)
                            print(train[u][j])
                            print(pr)
                            print(delta)
                            print(U[u])
                            print(V[j])

                # update U[u]
                U[u] = U[u] + eta * (user_gr / sum(idct[u]))  # sum(idct[u]) is the number of ratings of user u
                # print(user_gr, sum(idct[u]))
            # 本轮batch结束，升级V[j]
            for j in range(m):
                if sum(idct[batch_size * batch: batch_size * (batch + 1), j]) != 0:
                    try:
                        V[j] = V[j] + eta * (item_gr[j] / sum(idct[batch_size * batch: batch_size * (batch + 1), j]))
                    except Exception as e:
                        print("Error：", e)

        # 升级本次epoch结束后的目标函数
        try:
            loss = 1 / 2 * (np.sum(np.multiply(idct, (train - np.matmul(U, V.T))) ** 2)
                            + lambda_u * np.linalg.norm(U) ** 2 + lambda_v * np.linalg.norm(V) ** 2)
            # print("loss of %d-th is" % ii, loss)
        except Exception as e:
            print("%d-th epoch, error is:" % ii, e)
            print(np.linalg.norm(U))
            print(np.linalg.norm(V))

        losses.append(loss)
        rmse = RMSE(U, V, train)
        rmses.append(rmse)

        # if ii % 10 == 0:
            # print("%d-th epoch has finished" % ii)

    # 最后画出losses的图像
    # draw(losses, "Loss Function of Training Set")
    # draw([rmses], "RMSE of Training Set", range(1, maxEpoch + 1),
    #      "# of epoch", "rmse", ["pmf"])
    return U, V


def draw(lines, title, x, xlabel, ylabel, labels):
    """
    draw the diagram of loss function of PMF training process
    :param points: the lines of the diagram 也就是数据组的数量
    :param title: title
    x: xlabel name
    labels: label name
    :return: None
    """
    i = 0
    for l in lines:
        plt.plot(x, l, marker='o', label=labels[i])
        i += 1
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(title + ".png")
    plt.show()
    # plt.savefig()


def RMSE(U, V, data):
    """
    calculate the rmse of prediction and data
    :param U: users' latent matrix
    :param V: items' latent matrix
    :param data: the data set, maybe training set or test set
    :return: rmse
    """
    idct = np.where(data > 0, 1, 0)
    num = np.sum(idct)
    try:
        pre_out = np.matmul(U, V.T)
        np.where(pre_out > 5, 5, pre_out)
        np.where(pre_out < 0, 0, pre_out)
        delta = np.multiply(idct, pre_out - data)
    except Exception as e:
        print("In RMSE() warning:", e)
        print(np.matmul(U, V.T)[::100, ::100])

    rmse = np.sqrt(np.linalg.norm(delta) / num)
    return rmse


def t(data):
    """
    map the rating 1 <= x <= K to t(x) = (x - 1)/(K-1), 映射到[0, 1]
    :param data: the input data set
    :return: data set after mapping
    """
    np.where(data == 0, 0, (data - 1)/(5 - 1))
    return data


def g(x):
    """
    log函数，避免线性高斯模型做预测时超出评分范围
    :param x: 变量x
    :return: log函数计算后的值
    """
    return 1/(1 + np.exp(-x))

def testEta(train, test):
    """
    This funtion in charge of finding the best eta and lambda of PMF
    :param data: the training data
    :return:None
    """
    # 需要对测试集算rmse，对训练集算loss
    losses = []
    rmses = []
    etas = []
    eta = 0.2
    for i in range(0, 5):
        etas.append(str(eta))
        U, V, loss = trainPMF(train, 10, eta, 0, 0, 10, batch_size=100)
        losses.append(loss)
        # rmses.append(rmse)
        rmse = RMSE(U, V, test)
        rmses.append(rmse)
        eta = eta / 2
        print("%d round test of eta is done"% (i + 1))

    # draw diagrams for obj of train and RMSE of test
    draw(losses, "Obj of train set", range(1, 11), "Number of Epoch", "Obj", etas)
    draw([rmses], "RMSE of test set", etas, "The eta value", "RMSE", ["RMSE"])

def testLambda(train, test):
    """
    find the best value for lambda
    :param train: the training data set
    :param test: the test data set
    :return: None
    """
    losses = []
    rmses = []
    ldas = []
    lda = 1
    for i in range(6):
        ldas.append(lda)
        U, V, loss = trainPMF(train, 10, 0.1, lda, lda, 10, batch_size=100)
        losses.append(loss)
        rmse = RMSE(U, V, test)
        rmses.append(rmse)
        lda = lda / 2
        print("%dth round test of lambda is done" % (i + 1))

    # draw diagrams for obj of train and RMSE of test
    draw(losses, "Obj of train set", range(1, 11), "Number of Epoch", "Obj", ldas)
    draw([rmses], "RMSE of test set", ldas, "The eta value", "RMSE", ["RMSE"])


if __name__ == "__main__":
    print("This is an implementation of PMF")
