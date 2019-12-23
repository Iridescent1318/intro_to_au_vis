import numpy as np
import librosa.display
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from LogRegression import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    k = 10
    cv_num = 10

    if CROSS_VALID_MODE:
        acc_svm = np.zeros(cv_num)
        prec_svm = np.zeros(cv_num)
        recall_svm = np.zeros(cv_num)
        f1_svm = np.zeros(cv_num)
        acc_skt = np.zeros(cv_num)
        prec_skt = np.zeros(cv_num)
        recall_skt = np.zeros(cv_num)
        f1_skt = np.zeros(cv_num)

        for cn in range(cv_num):
            x_train, y_train, x_cvtest, y_cvtest = cross_validation(au_xs_train, au_ys_train, k)
            au_clf = svm.SVC(gamma='scale')
            au_clf.fit(x_train, y_train)
            skt_lr = LogisticRegression(random_state=0, solver='liblinear').fit(x_train, y_train)
            y_pred_svm = au_clf.predict(x_cvtest)
            y_pred_skt = skt_lr.predict(x_cvtest)

            print("Predictions of SVM: {}".format(y_pred_svm))
            print("Predictions of skt: {}".format(y_pred_skt))
            print("True:               {}".format(y_cvtest))

            acc_svm[cn] = accuracy_score(y_cvtest, y_pred_svm)
            prec_svm[cn] = precision_score(y_cvtest, y_pred_svm)
            recall_svm[cn] = recall_score(y_cvtest, y_pred_svm)
            f1_svm[cn] = f1_score(y_cvtest, y_pred_svm)
            acc_skt[cn] = accuracy_score(y_cvtest, y_pred_skt)
            prec_skt[cn] = precision_score(y_cvtest, y_pred_skt)
            recall_skt[cn] = recall_score(y_cvtest, y_pred_skt)
            f1_skt[cn] = f1_score(y_cvtest, y_pred_skt)

        print("SVM accuracy:  {}".format(acc_svm))
        print("SVM precision: {}".format(prec_svm))
        print("SVM recall:    {}".format(recall_svm))
        print("SVM F1-score : {}".format(f1_svm))
        print("skt accuracy:  {}".format(acc_skt))
        print("skt precision: {}".format(prec_skt))
        print("skt recall:    {}".format(recall_skt))
        print("skt F1-score : {}".format(f1_skt))
        print("SVM mean accuracy: {:.4f} precision: {:.4f} recall: {:.4f} f1-score:{:.4f}".format(
            np.mean(acc_svm), np.mean(prec_svm), np.mean(recall_svm), np.mean(acc_svm)))
        print("skt mean accuracy: {:.4f} precision: {:.4f} recall: {:.4f} f1-score:{:.4f}".format(
            np.mean(acc_skt), np.mean(prec_skt), np.mean(recall_skt), np.mean(acc_skt)))
    else:
        clf = svm.SVC(gamma='scale')
        clf.fit(au_xs_train, au_ys_train)
        y_pred = clf.predict(au_xs_test)
        accuracy = accuracy_score(au_ys_test, y_pred)
        precision = precision_score(au_ys_test, y_pred)
        recall = recall_score(au_ys_test, y_pred)
        f1 = f1_score(au_ys_test, y_pred)
        print("Prediction: {}".format(y_pred))
        print("True:       {}".format(au_ys_test))
        print("Accuracy:   {:.4f}".format(accuracy))
        print("Precision:  {:.4f}".format(accuracy))
        print("Recall:     {:.4f}".format(accuracy))
        print("F1-score  : {:.4f}".format(accuracy))
        # np.save("B.npy", y_pred)
