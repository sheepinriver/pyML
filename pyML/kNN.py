import numpy as np
from collections import Counter
from .metrics import accuracy_score


class KNNClassifier:

    def __init__(self, k):
        """初始化KNN分类器"""
        assert k >= 1, 'k must be valid'
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            'the size of X_train must be equal to the size of y_train'
        assert self.k <= X_train.shape[0], \
            'k must not exceed the size of X_train'

        self._X_train = X_train
        self._y_train = y_train
        return self

    def neighbour_indices(self, X_predict):
        """给定待预测数据机X_predict， 返回预测结果向量"""

        assert self._X_train is not None and self._y_train is not None, \
            'must fit before predict'
        assert X_predict.shape[1] == self._X_train.shape[1], \
            'the feature number of X_predict must be equal to X_train'

        return np.argsort(np.array([np.sqrt(np.sum((x_predict - self._X_train) ** 2, axis=1))
                                    for x_predict in X_predict]), axis=1)

    def predict(self, X_predict):
        neighbour_indices = self.neighbour_indices(X_predict)
        y_predict = np.array([Counter(y_predict_topK).most_common(1)[0][0]
                              for y_predict_topK in self._y_train[neighbour_indices[:, :self.k]]])
        return y_predict

    def score(self, X_test, y_test):
        """根据测试数据集X_test和y_test确定当前模型的准确度"""

        return accuracy_score(y_test, self.predict(X_test))

    def __repr__(self):
        return 'KNN(k={:d})'.format(self.k)
