import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据X 和 y 按照test_ratio分割成X_train, X, test, Y_train, Y_test"""

    assert X.shape[0] == y.shape[0], \
        'The size of X must be equal to the size of y'
    assert 0.0 <= test_ratio <= 1.0, \
        'test_ratio must be valid'

    if seed:
        np.random.seed(seed)

    shuffle_indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    train_indices = shuffle_indices[test_size:]
    test_indices = shuffle_indices[:test_size]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
