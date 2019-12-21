import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from LogRegression import score, cross_validation, LogRegression

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

    k = 4
    cv_num = 1
    max_epoch = 1000
    loss_visual_step = 10

    prec_bgd = np.zeros(cv_num)
    prec_newton = np.zeros(cv_num)
    prec_skt = np.zeros(cv_num)
    for cn in range(cv_num):
        x_train, y_train, x_cvtest, y_cvtest = cross_validation(im_xs_train, im_ys_train, k)
        lr_bgd = LogRegression(im_xs_train.shape[1])
        lr_newton = LogRegression(im_xs_train.shape[1])
        skt_lr = LogisticRegression(random_state=0, solver='liblinear').fit(x_train, y_train)
        loss_bgd = lr_bgd.fit(x_train, y_train, max_epoch, 0.18, 1e-4, 'bgd', loss_visual_step)
        loss_newton = lr_newton.fit(x_train, y_train, max_epoch, 0.18, 1e-4, 'newton', loss_visual_step)
        y_pred_bgd = lr_bgd.predict(x_cvtest)
        y_pred_newton = lr_newton.predict(x_cvtest)
        prec_bgd[cn] = score(y_pred_bgd, y_cvtest)
        prec_newton[cn] = score(y_pred_newton, y_cvtest)
        prec_skt[cn] = skt_lr.score(x_cvtest, y_cvtest)
        plt.plot((np.arange(len(loss_bgd)) + 1) * loss_visual_step, loss_bgd)
        plt.plot((np.arange(len(loss_newton)) + 1) * loss_visual_step, loss_newton)
        plt.legend(["BGD", "Newton"])

    print("sklearn LR accuracy:      {}".format(prec_skt))
    print("BGD Precision:            {}".format(prec_bgd))
    print("Newton Precision:         {}".format(prec_newton))
    print("Mean sklearn LR accuracy: {:.4f}".format(np.mean(prec_skt)))
    print("BGD mean precision:       {:.4f}".format(np.mean(prec_bgd)))
    print("Newton mean precision:    {:.4f}".format(np.mean(prec_newton)))
    plt.show()
