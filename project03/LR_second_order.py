import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from LogRegression import cross_validation, LogRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    k = 10
    cv_num = 10
    max_epoch = 500
    loss_visual_step = 10

    if CROSS_VALID_MODE:    
        acc = np.zeros(cv_num)
        prec = np.zeros(cv_num)
        recall = np.zeros(cv_num)
        f1 = np.zeros(cv_num)
        acc_skt = np.zeros(cv_num)
        prec_skt = np.zeros(cv_num)
        recall_skt = np.zeros(cv_num)
        f1_skt = np.zeros(cv_num)
        for cn in range(cv_num):
            x_train, y_train, x_cvtest, y_cvtest = cross_validation(im_xs_train, im_ys_train, k)
            lr = LogRegression(im_xs_train.shape[1])
            skt_lr = LogisticRegression(random_state=0, solver='newton-cg').fit(x_train, y_train)
            loss = lr.fit(x_train, y_train, max_epoch, 0.08, 1e-4, 'newton', loss_visual_step)
            
            y_pred = lr.predict(x_cvtest)
            acc[cn] = accuracy_score(y_cvtest, y_pred)
            prec[cn] = precision_score(y_cvtest, y_pred)
            recall[cn] = recall_score(y_cvtest, y_pred)
            f1[cn] = f1_score(y_cvtest, y_pred)

            y_pred_skt = skt_lr.predict(x_cvtest)
            acc_skt[cn] = accuracy_score(y_cvtest, y_pred_skt)
            prec_skt[cn] = precision_score(y_cvtest, y_pred_skt)
            recall_skt[cn] = recall_score(y_cvtest, y_pred_skt)
            f1_skt[cn] = f1_score(y_cvtest, y_pred_skt)
            plt.plot((np.arange(len(loss))+1) * loss_visual_step, loss)
    
        print("Accuracy:  {}".format(acc))
        print("Precision: {}".format(prec))
        print("Recall:    {}".format(recall))
        print("F1-score:  {}".format(f1))
        print("sklearn LR accuracy:  {}".format(acc_skt))
        print("           precision: {}".format(prec_skt))
        print("           recall:    {}".format(recall_skt))
        print("           F1-score:  {}".format(f1_skt))
        print("Mean sklearn LR accuracy: {:.4f} precision: {:.4f} recall: {:.4f} f1-score: {:.4f}".format(
            np.mean(acc_skt), np.mean(prec_skt), np.mean(recall_skt), np.mean(f1_skt)))
        print("Mean accuracy:            {:.4f} precision: {:.4f} recall: {:.4f} f1-score: {:.4f}".format(
            np.mean(acc), np.mean(prec), np.mean(recall), np.mean(f1)))
        plt.show()
        
    else:
        im_xs_test = np.ones((100, 13))
        im_ys_test = np.load("./test_result.npy")

        for i in range(100):
            im_xs_test[i, :] = np.load("./dataset/test/{}/feat.npy".format(i))

        lr = LogRegression(im_xs_train.shape[1])
        loss = lr.fit(im_xs_train, im_ys_train, max_epoch, l_rate=0.18, tol=1e-4, method='newton')
        y_pred_newton = lr.predict(im_xs_test)
        acc_newton = accuracy_score(im_ys_test, y_pred_newton)
        prec_newton = precision_score(im_ys_test, y_pred_newton)
        recall_newton = recall_score(im_ys_test, y_pred_newton)
        f1_newton = f1_score(im_ys_test, y_pred_newton)
        print(y_pred_newton)
        print(im_ys_test)

        skt_lr = LogisticRegression(random_state=0, solver='newton-cg').fit(im_xs_train, im_ys_train)
        y_pred_skt = skt_lr.predict(im_xs_test)
        acc_skt = accuracy_score(im_ys_test, y_pred_skt)
        prec_skt = precision_score(im_ys_test, y_pred_skt)
        recall_skt = recall_score(im_ys_test, y_pred_skt)
        f1_skt = f1_score(im_ys_test, y_pred_skt)
        print(y_pred_skt)
        print(im_ys_test)
        print("Newton method accuracy: {:.4f} precision: {:.4f} recall: {:.4f} f1-score: {:.4f}".format(
            acc_newton, prec_newton, recall_newton, f1_newton))
        print("          skt accuracy: {:.4f} precision: {:.4f} recall: {:.4f} f1-score: {:.4f}".format(
            acc_skt, prec_skt, recall_skt, f1_skt))

        np.save("A.npy", y_pred_skt)
