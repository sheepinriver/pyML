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
        assert X.shape[1] == len(self._mean), \
            'the feature of X must be equal to _mean and _scale'

        return (X - self._mean) / self._scale


class MinMaxScaler:

    def __init__(self):
        self._min = None
        self._max = None

    def fit(self, X):
        """根据训练数据集X获取最大和最小值"""

        assert X.ndim == 2, 'X must be 2 dimension'

        self._min = np.min(X, axis=0)
        self._max = np.max(X, axis=0)

        return self

    def transform(self, X):
        """将传入的X根据self._min和self._min进行最值归一化"""
        assert X.ndim == 2, 'X must be 2 dimension'
        assert self._min is not None and self._max is not None, \
            'must fit before transform'
        assert X.shape[1] == len(self._max), \
            'the feature of X must be equal to _min and _max'

        return (X - self._min) / (self._max - self._min)




