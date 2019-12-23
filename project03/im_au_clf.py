import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

CROSS_VALID_MODE = 1
TRAIN_MODE = 'ensemble'

if __name__ == '__main__':
    im_xs_train = np.ones((200, 13))
    im_ys_train = np.ones(200)

    for i in range(200):
        if i <= 99:
            im_xs_train[i, :] = np.load("./dataset/train/positive/{}/feat.npy".format(i))
            im_ys_train[i] = 1
        else:
            im_xs_train[i, :] = np.load("./dataset/train/negative/{}/feat.npy".format(i - 100))
            im_ys_train[i] = 0

    im_xs_test = np.ones((100, 13))

    for i in range(100):
        im_xs_test[i, :] = np.load("./dataset/test/{}/feat.npy".format(i))

    au_xs_train = np.load("au_xs_train.npy")
    au_ys_train = np.load("au_ys_train.npy")
    au_xs_test = np.load("au_xs_test.npy")

    im_au_xs_train = np.append(im_xs_train, au_xs_train, axis=1)
    im_au_xs_test = np.append(im_xs_test, au_xs_test, axis=1)
    im_au_ys_train = im_ys_train

    cv_num = 5
    test_size = 0.2

    if TRAIN_MODE == 'feat_combine':
        clf_num = 3
        if CROSS_VALID_MODE:
            acc = np.zeros(cv_num)
            pred = np.zeros(cv_num)
            recall = np.zeros(cv_num)
            f1 = np.zeros(cv_num)
            for cn in range(cv_num):
                clfs = []
                (ia_x_train, ia_x_test, ia_y_train, ia_y_test) = train_test_split(im_au_xs_train, im_au_ys_train,
                                                                                  test_size=test_size)

                for i in range(clf_num):
                    (x_train, _, y_train, _) = train_test_split(ia_x_train, ia_y_train, test_size=0.5)
                    lr = svm.SVC(gamma='scale')
                    lr.fit(x_train, y_train)
                    clfs.append(lr)

                y_pred_all = np.ones(ia_y_test.shape)
                for clf in clfs:
                    y_pred_all += clf.predict(ia_x_test)

                y_pred_all = np.where(y_pred_all > clf_num / 2, 1, 0)
                acc[cn] = accuracy_score(ia_y_test, y_pred_all)
                pred[cn] = precision_score(ia_y_test, y_pred_all)
                recall[cn] = recall_score(ia_y_test, y_pred_all)
                f1[cn] = f1_score(ia_y_test, y_pred_all)

            print("Accuracy:            {}".format(acc))
            print("Precision:           {}".format(pred))
            print("Recall:              {}".format(recall))
            print("F1:                  {}".format(f1))
            print("Mean accuracy:   {:.4f}".format(np.mean(acc)))
            print("Mean precision:  {:.4f}".format(np.mean(pred)))
            print("Mean recall   :  {:.4f}".format(np.mean(recall)))
            print("Mean f1-score:   {:.4f}".format(np.mean(f1)))

        else:
            clfs = []
            for i in range(clf_num):
                (x_train, _, y_train, _) = train_test_split(im_au_xs_train, im_au_ys_train, test_size=0.5)
                lr = svm.SVC(gamma='scale')
                lr.fit(x_train, y_train)
                clfs.append(lr)

            y_pred_all = np.ones(100)
            for clf in clfs:
                y_pred_all += clf.predict(im_au_xs_test)

            y_pred_all = np.where(y_pred_all > clf_num / 2, 1, 0)

            print("Predict: {}".format(y_pred_all))

            # np.save("C.npy", y_pred_all)

    if TRAIN_MODE == 'ensemble':
        if CROSS_VALID_MODE:
            acc = np.zeros(cv_num)
            pred = np.zeros(cv_num)
            recall = np.zeros(cv_num)
            f1 = np.zeros(cv_num)
            for cn in range(cv_num):
                (x_train, x_test, y_train, y_test) = train_test_split(im_au_xs_train, im_au_ys_train, test_size=test_size)
                im_clf = LogisticRegression(random_state=0, solver='newton-cg')
                im_clf.fit(x_train[:, 0:13], y_train)
                y_pred_im = im_clf.predict_proba(x_test[:, 0:13])[:, 1]

                au_clf = svm.SVC(gamma='scale', probability=True)
                au_clf.fit(x_train[:, 13:], y_train)
                y_pred_au = au_clf.predict_proba(x_test[:, 13:])[:, 1]

                y_pred_all = y_pred_im + y_pred_au
                y_pred_all = np.where(y_pred_all >= 1, 1, 0)

                acc[cn] = accuracy_score(y_test, y_pred_all)
                pred[cn] = precision_score(y_test, y_pred_all)
                recall[cn] = recall_score(y_test, y_pred_all)
                f1[cn] = f1_score(y_test, y_pred_all)

            print("Accuracy:            {}".format(acc))
            print("Precision:           {}".format(pred))
            print("Recall:              {}".format(recall))
            print("F1:                  {}".format(f1))
            print("Mean accuracy:   {:.4f}".format(np.mean(acc)))
            print("Mean precision:  {:.4f}".format(np.mean(pred)))
            print("Mean recall   :  {:.4f}".format(np.mean(recall)))
            print("Mean f1-score:   {:.4f}".format(np.mean(f1)))

        else:
            im_clf = LogisticRegression(random_state=0, solver='newton-cg')
            im_clf.fit(im_xs_train, im_ys_train)
            y_pred_im = im_clf.predict_proba(im_xs_test)[:, 1]

            au_clf = svm.SVC(gamma='scale', probability=True)
            au_clf.fit(au_xs_train, au_ys_train)
            y_pred_au = au_clf.predict_proba(au_xs_test)[:, 1]

            y_pred_all = y_pred_im + y_pred_au
            y_pred_all = np.where(y_pred_all >= 1, 1, 0)

            print("Predict: {}".format(y_pred_all))

            # np.save("C.npy", y_pred_all)
