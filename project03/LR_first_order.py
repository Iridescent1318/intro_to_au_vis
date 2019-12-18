import sklearn
import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


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
            if dp + self.bias > 0:
                return 1 / (1 + math.exp(- dp - self.bias))
            else:
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
                loss = 0
                for j in range(xs.shape[0]):
                    y_hat = self.get_sigmoid(xs[j, :])
                    w_grad += (y_hat - ys[j]) * xs[j, :]
                    b_grad += (y_hat - ys[j])
                self.weights = self.weights - l_rate * w_grad
                self.bias = self.bias - l_rate * b_grad
                for j in range(xs.shape[0]):
                    dp = np.dot(xs[j, :], self.weights)
                    if dp + self.bias > 10:
                        loss += (dp + self.bias) * (1 - ys[j])
                    else:
                        loss += math.log(1+math.exp(dp + self.bias)) - (dp + self.bias) * ys[j]
                return w_grad, b_grad, loss
            else:
                raise Exception("Dimension of x and weights should be equal.")
        else:
            raise Exception("Length of x's and y's should be equal.")
        pass

    def fit(self, epoch_num, l_rate, visualize=False, vis_epoch_step=10):
        for en in range(epoch_num):
            w_grad, b_grad, loss = lr.batch_grad_descent(xs_train, ys_train, l_rate)
            if visualize:
                if en % vis_epoch_step == 0:
                    print("--------------------------------Epoch:{}--------------------------------".format(en))
                    # print("weights: {} \n bias:   {}".format(self.weights, self.bias))
                    # print("w_grad:  {} \n b_grad: {}".format(w_grad, b_grad))
                    print("loss: {}".format(loss))
                    print("------------------------------------------------------------------------")

    def predict(self, xs, proba=False):
        if xs.shape[1] == self.weights.shape[0]:
            y_predict = np.zeros(xs.shape[0])
            for j in range(xs.shape[0]):
                if proba:
                    y_predict[j] = self.get_sigmoid(xs[j])
                else:
                    y_predict[j] = 1 if self.get_sigmoid(xs[j]) >= 0.5 else 0
            return y_predict
        else:
            raise Exception("Dimension of x and weights should be equal.")


def cross_validation(xs, ys, k):
    if xs.shape[0] == ys.shape[0]:
        if xs.shape[0] % k == 0:
            test_num = int(xs.shape[0] / k)
            shuffle_index = np.arange(xs.shape[0])
            np.random.shuffle(shuffle_index)
            xs_shuffled = xs[shuffle_index]
            ys_shuffled = ys[shuffle_index]
            xs_cvtest = xs_shuffled[(xs.shape[0] - test_num):xs.shape[0], :]
            ys_cvtest = ys_shuffled[(xs.shape[0] - test_num):xs.shape[0]]
            xs_train = xs_shuffled[0:(xs.shape[0] - test_num), :]
            ys_train = ys_shuffled[0:(xs.shape[0] - test_num)]
            return xs_train, ys_train, xs_cvtest, ys_cvtest
        else:
            raise Exception("Number of xs can't be divided by k.")
    else:
        raise Exception("Length of x's and y's should be equal.")


def score(y_predict, y_real):
    if y_predict.shape == y_real.shape:
        test_num = y_predict.shape[0]
        precision = 0
        for yp, yr in zip(y_predict, y_real):
            precision += 1 if yp == yr else 0
        precision /= test_num
        return precision
    else:
        raise Exception("y_predict, y_real are not in the same shape.")


chara_vectors_train = np.ones((200, 13))
chara_results_train = np.ones(200)

for i in range(200):
    if i <= 99:
        chara_vectors_train[i, :] = np.load("./dataset/train/positive/{}/feat.npy".format(i))
        chara_results_train[i] = 1
    else:
        chara_vectors_train[i, :] = np.load("./dataset/train/negative/{}/feat.npy".format(i-100))
        chara_results_train[i] = 0

k = 8
cv_num = 10
epoch_num = 300

prec = np.zeros(cv_num)
prec_skt = np.zeros(cv_num)
for cn in range(cv_num):
    xs_train, ys_train, xs_cvtest, ys_cvtest = cross_validation(chara_vectors_train, chara_results_train, k)
    lr = LogRegression(chara_vectors_train.shape[1])
    skt_lr = LogisticRegression(random_state=0, solver='lbfgs').fit(xs_train, ys_train)
    lr.fit(epoch_num, 0.5, False)
    y_pred = lr.predict(xs_cvtest)
    prec[cn] = score(y_pred, ys_cvtest)
    prec_skt[cn] = skt_lr.score(xs_cvtest, ys_cvtest)

print("sklearn LR accuracy: {}".format(prec_skt))
print("Precision:           {}".format(prec))
