import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
import os

FEAT_CLF_MODE = 0

if __name__ == '__main__':
    if FEAT_CLF_MODE:
        emotion_xs_train = []
        emotion_ys_train = []
        emotion_xs_test_idx = []
        emotion_xs_test = []

        for i in range(200):
            for j in range(30):
                if i <= 99:
                    if os.path.exists("./dataset/train/positive/{}/{}_emotions.npy".format(i, j)):
                        emo = np.load("./dataset/train/positive/{}/{}_emotions.npy".format(i, j)).tolist()
                        emotion_xs_train.append(emo)
                        emotion_ys_train.append(1)
                else:
                    if os.path.exists("./dataset/train/negative/{}/{}_emotions.npy".format(i-100, j)):
                        emo = np.load("./dataset/train/negative/{}/{}_emotions.npy".format(i-100, j)).tolist()
                        emotion_xs_train.append(emo)
                        emotion_ys_train.append(0)

        emotion_xs_train = np.array(emotion_xs_train)
        emotion_ys_train = np.array(emotion_ys_train)

        for i in range(100):
            for j in range(30):
                if os.path.exists("./dataset/test/{}/{}_emotions.npy".format(i, j)):
                    emo = np.load("./dataset/test/{}/{}_emotions.npy".format(i, j)).tolist()
                    emotion_xs_test_idx.append(i)
                    emotion_xs_test.append(emo)

        emotion_xs_test = np.array(emotion_xs_test)
        emotion_xs_test_idx = np.array(emotion_xs_test_idx)

        np.save("emotion_xs_train.npy", emotion_xs_train)
        np.save("emotion_ys_train.npy", emotion_ys_train)
        np.save("emotion_xs_test_idx.npy", emotion_xs_test_idx)
        np.save("emotion_xs_test.npy", emotion_xs_test)
    else:
        emotion_xs_train = np.load("emotion_xs_train.npy")
        emotion_ys_train = np.load("emotion_ys_train.npy")
        emotion_xs_test_idx = np.load("emotion_xs_test_idx.npy")
        emotion_xs_test = np.load("emotion_xs_test.npy")
        emotion_ys_test = np.load("test_result.npy")

        clf = svm.SVC(gamma='scale')
        clf.fit(emotion_xs_train, emotion_ys_train)
        emotions_pred = clf.predict(emotion_xs_test)
        emotion_ys_pred = np.zeros(emotion_ys_test.shape)
        emotion_ys_idxnum = np.zeros(emotion_ys_test.shape)
        for idx, ep in zip(emotion_xs_test_idx, emotions_pred):
            emotion_ys_pred[idx] += ep
            emotion_ys_idxnum[idx] += 1
        emotion_ys_pred = np.true_divide(emotion_ys_pred, emotion_ys_idxnum)
        emotion_ys_pred = np.where(emotion_ys_pred >= 0.5, 1, 0)
        accuracy = accuracy_score(emotion_ys_test, emotion_ys_pred)
        print(accuracy)
