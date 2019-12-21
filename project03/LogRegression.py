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
            if dp + self.bias > 0:
                return 1 / (1 + math.exp(- dp - self.bias))
            else:
                return math.exp(dp + self.bias) / (1 + math.exp(dp + self.bias))
        else:
            raise Exception("Not in the same shape. Object weight shape:{} Input shape: {}".format(self.weights.shape,
                                                                                                   x.shape))
        pass

    def get_loss(self, xs, ys):
        if xs.shape[1] == self.weights.shape[0]:
            loss = 0
            for j in range(xs.shape[0]):
                dp = np.dot(xs[j, :], self.weights)
                if dp + self.bias > 10:
                    loss += (dp + self.bias) * (1 - ys[j])
                else:
                    loss += math.log(1 + math.exp(dp + self.bias)) - (dp + self.bias) * ys[j]
            return loss
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
                w_grad /= xs.shape[0]
                b_grad /= xs.shape[0]
                self.weights = self.weights - l_rate * w_grad
                self.bias = self.bias - l_rate * b_grad
                for j in range(xs.shape[0]):
                    dp = np.dot(xs[j, :], self.weights)
                    if dp + self.bias > 20:
                        loss += (dp + self.bias) * (1 - ys[j])
                    else:
                        loss += math.log(1+math.exp(dp + self.bias)) - (dp + self.bias) * ys[j]
                return loss
            else:
                raise Exception("Dimension of x and weights should be equal.")
        else:
            raise Exception("Length of x's and y's should be equal.")

    def newton_method(self, xs, ys, l_rate):
        if xs.shape[0] == ys.shape[0]:
            if xs.shape[1] == self.weights.shape[0]:
                w_grad = np.zeros(self.weights.shape)
                w_hessian = np.zeros((xs.shape[1], xs.shape[1]))
                b_grad = 0
                b_hessian = 0
                for j in range(xs.shape[0]):
                    y_hat = self.get_sigmoid(xs[j, :])
                    w_grad += (y_hat - ys[j]) * xs[j, :]
                    w_hessian += y_hat * (1 - y_hat) * np.matmul(xs[j, :].T, xs[j, :])
                    b_grad += (y_hat - ys[j])
                    b_hessian += y_hat * (1 - y_hat)
                w_grad /= xs.shape[0]
                b_grad /= xs.shape[0]
                w_hessian /= xs.shape[0]

                if np.linalg.det(w_hessian):
                    w_hessian_inv = np.linalg.inv(w_hessian)
                else:
                    w_hessian += np.eye(xs.shape[1])
                    w_hessian_inv = np.linalg.inv(w_hessian)

                self.weights += -l_rate * np.matmul(w_grad, w_hessian_inv.T)
                self.bias += -l_rate * b_grad / b_hessian
                loss = self.get_loss(xs, ys)
                return loss
            else:
                raise Exception("Dimension of x and weights should be equal.")
        else:
            raise Exception("Length of x's and y's should be equal.")

    def fit(self, xs, ys, max_epoch_num=500, l_rate=1e-2, tol=1e-3, method='bgd', loss_vis_step=10, visualize=False, vis_epoch_step=10):
        prev_loss = 0
        loss_list = []
        self.set_weights(np.random.normal(size=self.weights.shape))
        for en in range(max_epoch_num):
            if method == 'bgd':
                cur_loss = self.batch_grad_descent(xs, ys, l_rate)
            else:
                if method == 'newton':
                    cur_loss = self.newton_method(xs, ys, l_rate)
                else:
                    raise Exception("Invalid method")
            if en % loss_vis_step == 0 and en:
                loss_list.append(cur_loss)
            if prev_loss and math.fabs(prev_loss-cur_loss)/cur_loss < tol:
                break
            prev_loss = cur_loss

            if visualize:
                if en % vis_epoch_step == 0:
                    print("--------------------------------Epoch:{}--------------------------------".format(en))
                    print("weights: {} \n bias:   {}".format(self.weights, self.bias))
                    print("loss: {}".format(cur_loss))
                    print("------------------------------------------------------------------------")
        return loss_list

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
