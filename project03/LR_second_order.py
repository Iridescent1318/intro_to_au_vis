import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from LogRegression import score, cross_validation, LogRegression

if __name__ == '__main__':
    chara_vectors_train = np.ones((200, 13))
    chara_results_train = np.ones(200)

    for i in range(200):
        if i <= 99:
            chara_vectors_train[i, :] = np.load("./dataset/train/positive/{}/feat.npy".format(i))
            chara_results_train[i] = 1
        else:
            chara_vectors_train[i, :] = np.load("./dataset/train/negative/{}/feat.npy".format(i-100))
            chara_results_train[i] = 0

    k = 8
    cv_num = 10
    epoch_num = 500
    loss_visual_step = 10

    prec = np.zeros(cv_num)
    prec_skt = np.zeros(cv_num)
    for cn in range(cv_num):
        x_train, y_train, x_cvtest, y_cvtest = cross_validation(chara_vectors_train, chara_results_train, k)
        lr = LogRegression(chara_vectors_train.shape[1])
        skt_lr = LogisticRegression(random_state=0, solver='newton-cg').fit(x_train, y_train)
        loss = lr.fit(x_train, y_train, epoch_num, 0.08, 1e-4, 'newton', loss_visual_step)
        y_pred = lr.predict(x_cvtest)
        prec[cn] = score(y_pred, y_cvtest)
        prec_skt[cn] = skt_lr.score(x_cvtest, y_cvtest)
        plt.plot((np.arange(len(loss))+1) * loss_visual_step, loss)

    print("sklearn LR accuracy: {}".format(prec_skt))
    print("Precision:           {}".format(prec))
    print("Mean sklearn LR accuracy: {:.4f}±{:.4f}".format(np.mean(prec_skt), np.std(prec_skt)))
    print("Mean precision:           {:.4f}±{:.4f}".format(np.mean(prec), np.std(prec)))
    plt.show()
