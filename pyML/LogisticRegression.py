import numpy as np
from .metrics import accuracy_score


class LogisticRegression:

    def __int__(self):
        """初始化Logistic Regression"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def _J(self, theta, X_b, y):
        y_hat = self._sigmoid(X_b.dot(theta))

        try:
            return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
        except:
            return float('inf')

    def _dJ(self, theta, X_b, y):
        return (self._sigmoid(X_b.dot(theta)) - y).dot(X_b) / len(X_b)

    def _gradient_descent(self, X_b, y, initial_theta, eta, n_iterations=10000, epsilon=1e-8):

        theta = initial_theta

        for i in range(n_iterations):
            gradient = self._dJ(theta, X_b, y)
            last_theta = theta
            theta = theta - eta * gradient
            if abs(self._J(theta, X_b, y) - self._J(last_theta, X_b, y)) < epsilon:
                break

        return theta

    def fit(self, X_train, y_train, eta=0.01, n_iterations=10000):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Logistic Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones(shape=(len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = self._gradient_descent(X_b, y_train, initial_theta, eta, n_iterations)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict_probability(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果概率向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones(shape=(len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        probability = self.predict_probability(X_predict)

        return np.array(probability >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"
