import numpy as np
from LogRegression import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

DIY_ENSEMBLE = 1

if __name__ == '__main__':
    au_xs_train = np.load("au_xs_train.npy")
    au_ys_train = np.load("au_ys_train.npy")
    au_xs_test = np.load("au_xs_test.npy")
    au_ys_test = np.load("./test_result.npy")

    k = 5
    cv_num = 5
    max_epoch = 500
    loss_visual_step = 10

    if DIY_ENSEMBLE:
        clfs = []
        clf_num = 20

        for i in range(clf_num):
            (x_train, _, y_train, _) = train_test_split(au_xs_train, au_ys_train, test_size=0.2)
            lr = svm.SVC(gamma='scale')
            lr.fit(x_train, y_train)
            clfs.append(lr)

        y_pred_all = np.ones(100)
        for clf in clfs:
            y_pred_all += clf.predict(au_xs_test)

        y_pred_all = np.where(y_pred_all > clf_num/2, 1, 0)

        print("Predict: {}".format(y_pred_all))
        print("True:    {}".format(au_ys_test))

        accuracy = accuracy_score(y_pred_all, au_ys_test)

        print("accuracy: {}".format(accuracy))
    else:
        clf = AdaBoostClassifier(n_estimators=150, learning_rate=1).fit(au_xs_train, au_ys_train)
        y_pred = clf.predict(au_xs_test)

        print("Predict: {}".format(y_pred))
        print("True:    {}".format(au_ys_test))

        accuracy = accuracy_score(y_pred, au_ys_test)

        print("accuracy: {}".format(accuracy))
