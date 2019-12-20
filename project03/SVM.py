import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn import svm
from LogRegression import cross_validation, score

FEAT_EXT_MODE = 0

if __name__ == '__main__':
    num_mfcc = 1
    if FEAT_EXT_MODE:
        au_test, au_sr = librosa.load("./dataset/train/positive/0/audio.wav")
        au_test = librosa.feature.mfcc(au_test, au_sr, n_mfcc=num_mfcc).reshape(-1)
        au_xs_train = np.ones((200, au_test.shape[0]))
        au_ys_train = np.ones(200)

        for i in range(200):
            if i <= 99:
                au, sr = librosa.load("./dataset/train/positive/{}/audio.wav".format(i))
                au_mfcc = librosa.feature.mfcc(au, sr, n_mfcc=num_mfcc).reshape(-1)
                au_xs_train[i, :] = au_mfcc
                au_ys_train[i] = 1
            else:
                au, sr = librosa.load("./dataset/train/negative/{}/audio.wav".format(i-100))
                au_mfcc = librosa.feature.mfcc(au, sr, n_mfcc=num_mfcc).reshape(-1)
                au_xs_train[i, :] = librosa.feature.mfcc(au, sr).reshape(-1)
                au_ys_train[i] = 0

        np.save("au_feat_xs.npy", au_xs_train)
        np.save("au_feat_ys.npy", au_ys_train)
    else:
        au_xs_train = np.load("au_feat_xs.npy")
        au_ys_train = np.load("au_feat_ys.npy")

    k = 2
    cv_num = 1

    for cn in range(cv_num):
        x_train, y_train, x_cvtest, y_cvtest = cross_validation(au_xs_train, au_ys_train, k)
        au_clf = svm.SVC()
        au_clf.fit(x_train, y_train)
        y_pred = au_clf.predict(x_cvtest)
        print("Predictions: {}".format(y_pred))
        print("Real:        {}".format(y_cvtest))
        precision = score(y_pred, y_cvtest)
        print("Precision: {}".format(precision))
