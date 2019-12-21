import numpy as np
import os

if __name__ == '__main__':
    test = np.ones((1, 30, 7))
    for i in range(30):
        if os.path.exists("./dataset/train/positive/0/{}_emotions.npy".format(i)):
            test[0, i, :] = np.load("./dataset/train/positive/0/{}_emotions.npy".format(i))
    print(test)
