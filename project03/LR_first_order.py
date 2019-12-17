import sklearn
import numpy as np
import math


class LogRegression:
    dim = 0
    weights = []
    bias = 0

    def __init__(self, dim, bias=0):
        self.dim = dim
        self.weights = np.zeros(dim)
        self.bias = bias

    def set_weights(self, weight_vec):
        if weight_vec.shape == self.weights.shape:
            self.weights = weight_vec
        else:
            raise Exception("Not in the same shape. Object weight shape:{} Input shape: {}".format(self.weights.shape,
                                                                                                   weight_vec.shape))

    def get_sigmoid(self, x):
        if x.shape == self.weights.shape:
            dp = np.dot(x, self.weights)
            return math.exp(dp + self.bias) / (1 + math.exp(dp + self.bias))
        else:
            raise Exception("Not in the same shape. Object weight shape:{} Input shape: {}".format(self.weights.shape,
                                                                                                   x.shape))
        pass

    def batch_grad_descent(self, xs, ys, l_rate):
        if xs.shape[0] == ys.shape[0]:
            if xs.shape[1] == self.weights.shape[0]:
                w_grad = np.zeros(self.weights.shape)
                b_grad = 0
                for j in range(xs.shape[0]):
                    y_hat = self.get_sigmoid(xs[j, :])
                    w_grad += (y_hat - ys[j]) * xs[j, :]
                    b_grad += (y_hat - ys[j])
                self.weights = self.weights - l_rate * w_grad
                self.bias = self.bias - l_rate * b_grad
            else:
                raise Exception("Dimension of x and weights should be equal.")
        else:
            raise Exception("Length of x's and y's should be equal.")
        pass


chara_vectors_train = np.ones((200, 13))
chara_results_train = np.ones(200)

for i in range(200):
    if i <= 99:
        chara_vectors_train[i, :] = np.load("./dataset/train/positive/{}/feat.npy".format(i))
        chara_results_train[i] = 1
    else:
        chara_vectors_train[i, :] = np.load("./dataset/train/negative/{}/feat.npy".format(i-100))
        chara_results_train[i] = 0

shuffle_index = np.arange(200)
np.random.shuffle(shuffle_index)
chara_vectors_train = chara_vectors_train[shuffle_index]
chara_results_train = chara_results_train[shuffle_index]

test_num = 100
epoch_num = 100

chara_vectors_cvtest = chara_vectors_train[(200-test_num):200, :]
chara_results_cvtest = chara_results_train[(200-test_num):200]
chara_vectors_train = chara_vectors_train[0:(200-test_num), :]
chara_results_train = chara_results_train[0:(200-test_num)]

lr = LogRegression(13)

for i in range(epoch_num):
    lr.batch_grad_descent(chara_vectors_train, chara_results_train, 1e-2)

precision = 0


for i in range(test_num):
    res = lr.get_sigmoid(chara_vectors_cvtest[i, :])
    res_bin = 1 if res >= 0.5 else 0
    if res_bin == chara_results_cvtest[i]:
        precision += 1
    print("Index:{} Estimation:{} Real:{}".format(i, res, chara_results_cvtest[i]))

precision /= test_num
print("Precision: {}".format(precision))
