import numpy as np


def accuracy_score(y_true, y_predict):
    """计算y_true与y_predict之间的准确率"""

    assert y_true.shape[0] == y_predict.shape[0], \
        'The size of y_true must be equal to the size of y_predict'

    return np.sum(y_true == y_predict) / len(y_true)

