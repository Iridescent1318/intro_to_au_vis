import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from LogRegression import accuracy_score, cross_validation, LogRegression

CROSS_VALID_MODE = 0

if __name__ == '__main__':
    im_xs_train = np.ones((200, 13))
    im_ys_train = np.ones(200)

    for i in range(200):
        if i <= 99:
            im_xs_train[i, :] = np.load("./dataset/train/positive/{}/feat.npy".format(i))
            im_ys_train[i] = 1
        else:
            im_xs_train[i, :] = np.load("./dataset/train/negative/{}/feat.npy".format(i-100))
            im_ys_train[i] = 0

    k = 8
    cv_num = 1
    max_epoch = 500
    loss_visual_step = 10

    if CROSS_VALID_MODE:    
        acc = np.zeros(cv_num)
        acc_skt = np.zeros(cv_num)
        for cn in range(cv_num):
            x_train, y_train, x_cvtest, y_cvtest = cross_validation(im_xs_train, im_ys_train, k)
            lr = LogRegression(im_xs_train.shape[1])
            skt_lr = LogisticRegression(random_state=0, solver='newton-cg').fit(x_train, y_train)
            loss = lr.fit(x_train, y_train, max_epoch, 0.08, 1e-4, 'newton', loss_visual_step)
            y_pred = lr.predict(x_cvtest)
            acc[cn] = accuracy_score(y_pred, y_cvtest)
            acc_skt[cn] = skt_lr.score(x_cvtest, y_cvtest)
            plt.plot((np.arange(len(loss))+1) * loss_visual_step, loss)
    
        print("sklearn LR accuracy: {}".format(acc_skt))
        print("Accuracy:           {}".format(acc))
        print("Mean sklearn LR accuracy: {:.4f}".format(np.mean(acc_skt)))
        print("Mean accuracy:           {:.4f}".format(np.mean(acc)))
        plt.show()
    else:
        im_xs_test = np.ones((100, 13))
        im_ys_test = np.load("./test_result.npy")

        for i in range(100):
            im_xs_test[i, :] = np.load("./dataset/test/{}/feat.npy".format(i))

        lr = LogRegression(im_xs_train.shape[1])
        loss = lr.fit(im_xs_train, im_ys_train, max_epoch, l_rate=0.18, tol=1e-4, method='newton')
        y_pred_newton = lr.predict(im_xs_test)
        acc_bgd = accuracy_score(y_pred_newton, im_ys_test)
        print(y_pred_newton)
        print(im_ys_test)
        print("Newton method accuracy: {}".format(acc_bgd))

        skt_lr = LogisticRegression(random_state=0, solver='newton-cg').fit(im_xs_train, im_ys_train)
        y_pred_skt = skt_lr.predict(im_xs_test)
        acc_skt = skt_lr.score(im_xs_test, im_ys_test)
        print(y_pred_skt)
        print(im_ys_test)
        print("skl accuracy: {}".format(acc_skt))
        np.save("A_2nd_order.npy", y_pred_newton)
