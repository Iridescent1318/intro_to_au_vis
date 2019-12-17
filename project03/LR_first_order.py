import sklearn
import numpy as np
import math


class LogRegression:
    dim = 0
    weights = []
    bias = 0

    def __init__(self, dim, bias=0):
        self.dim = dim
        self.weights = np.ones(dim)
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
            return 1 / (1 + math.exp(-dp - self.bias))
        else:
            raise Exception("Not in the same shape. Object weight shape:{} Input shape: {}".format(self.weights.shape,
                                                                                                   x.shape))
        pass

    def BGD(self, xs):
        pass


chara_vectors = np.ones((200, 13))

test2 = np.ones((5, 1))

for i in range(200):
    if i <= 99:
        chara_vectors[i, :] = np.load("./dataset/train/positive/{}/feat.npy".format(i))
    else:
        chara_vectors[i, :] = np.load("./dataset/train/negative/{}/feat.npy".format(i-100))



print(test2.shape)
print(test2)