import numpy as np
import librosa.display
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from LogRegression import cross_validation, score

FEAT_EXT_MODE = 0
CROSS_VALID_MODE = 0

if __name__ == '__main__':
    num_mfcc = 1
    if FEAT_EXT_MODE:
        au_test, au_sr = librosa.load("./dataset/train/positive/0/audio.wav")
        au_test = librosa.feature.mfcc(au_test, au_sr, n_mfcc=num_mfcc).reshape(-1)
        au_xs_train = np.ones((200, au_test.shape[0]))
        au_xs_test = np.ones((100, au_test.shape[0]))
        au_ys_train = np.ones(200)
        au_ys_test = np.load("./test_result.npy")

        for i in range(200):
            if i <= 99:
                au, sr = librosa.load("./dataset/train/positive/{}/audio.wav".format(i))
                au_mfcc = librosa.feature.mfcc(au, sr, n_mfcc=num_mfcc).reshape(-1)
                au_xs_train[i, :] = au_mfcc
                au_ys_train[i] = 1
            else:
                au, sr = librosa.load("./dataset/train/negative/{}/audio.wav".format(i-100))
                au_mfcc = librosa.feature.mfcc(au, sr, n_mfcc=num_mfcc).reshape(-1)
                au_xs_train[i, :] = au_mfcc
                au_ys_train[i] = 0

        for i in range(100):
            au, sr = librosa.load("./dataset/test/{}/audio.wav".format(i))
            au_mfcc = librosa.feature.mfcc(au, sr, n_mfcc=num_mfcc).reshape(-1)
            au_xs_test[i, :] = au_mfcc

        np.save("au_xs_train.npy", au_xs_train)
        np.save("au_ys_train.npy", au_ys_train)
        np.save("au_xs_test.npy", au_xs_test)

    else:
        au_xs_train = np.load("au_xs_train.npy")
        au_ys_train = np.load("au_ys_train.npy")
        au_xs_test = np.load("au_xs_test.npy")
        au_ys_test = np.load("./test_result.npy")

    k = 2
    cv_num = 10

    if CROSS_VALID_MODE:
        prec_svm = np.zeros(cv_num)
        prec_skt = np.zeros(cv_num)

        for cn in range(cv_num):
            x_train, y_train, x_cvtest, y_cvtest = cross_validation(au_xs_train, au_ys_train, k)
            au_clf = svm.SVC(gamma='scale')
            au_clf.fit(x_train, y_train)
            skt_lr = LogisticRegression(random_state=0, solver='liblinear').fit(x_train, y_train)
            y_pred_svm = au_clf.predict(x_cvtest)
            y_pred_skt = skt_lr.predict(x_cvtest)
            print("Predictions of SVM: {}".format(y_pred_svm))
            print("Predictions of skt: {}".format(y_pred_skt))
            print("Real:               {}".format(y_cvtest))
            prec_svm[cn] = score(y_pred_svm, y_cvtest)
            prec_skt[cn] = score(y_pred_skt, y_cvtest)

        print("Precision of SVM: {}".format(prec_svm))
        print("Precision of skt: {}".format(prec_skt))
        print("Mean Precision of SVM: {}".format(np.mean(prec_svm)))
        print("Mean Precision of skt: {}".format(np.mean(prec_skt)))
    else:
        clf = svm.SVC(gamma='scale')
        clf.fit(au_xs_train, au_ys_train)
        y_pred = clf.predict(au_xs_test)
        precision = clf.score(au_xs_test, au_ys_test)
        print("Precision: {}".format(precision))
        print("Prediction: {}".format(y_pred))
        print("Real:       {}".format(au_ys_test))
        np.save("B.npy", y_pred)
