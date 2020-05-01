import numpy as np


class StandardScaler:

    def __init__(self):
        self._mean = None
        self._scale = None

    def fit(self, X):
        """根据训练数据集X获得数据的均值和方差"""
        assert X.ndim == 2, 'X must be 2 dimension'

        self._mean = np.mean(X, axis=0)
        self._scale = np.std(X, axis=0)

        return self

    def transform(self, X):
        """将传入的X根据self._mean和self._scale进行均值方差归一化"""
        assert X.ndim == 2, 'X must be 2 dimension'
        assert self._mean is not None and self._scale is not None, \
            'must fit before transform'

        return (X - self._mean) / self._scale


