import numpy as np
import matplotlib.pyplot as plt
from .kNN import KNNClassifier

def draw_2D_kNN(X, y, X_predict, k=5):
    """画一个二维的散点图和折线图，用以演示kNN算法"""

    assert X.shape[1] == X_predict[1] == 2, \
        'X and x_predict must have two features '
    assert X.shape[0] == y.shape[0], \
        'The size of X must be equal to the size of y'
    assert k >= 1, 'k must be valid'

    # 实例一个画布 - 当只有一个图的时候，不是必须的
    plt.figure(figsize=(16, 12))

    y_categories = np.unique(y)

    for y_category in y_categories:
        plt.scatter(X[y == y_category, 0], X[y == y_category, 1],
                    c=y_category,
                    label='Category {:s}'.format(str(y_category))
                    )

    kNN_classifier = KNNClassifier(k)
    kNN_classifier.fit(X, y)
    neighbour_indices = kNN_classifier.neighbour_indices(X_predict)

    for i in len(X_predict):
        for j in range(k):
            # 每次循环构造两个点
            plot_x = [X_predict[i, 0], X[neighbour_indices[i], 0]]
            plot_y = [X_predict[i, 1], X[neighbour_indices[i], 1]]
    # 画两点之间点连线
    plt.plot(plot_x, plot_y, color='r')


    plt.tick_params(direction="out"
                    , length=6
                    , width=2
                    , colors="w"
                    # , grid_color='r'
                    # , grid_alpha=0.5
                    )
    plt.legend()
    plt.title('kNN Classifier', color='w')
    plt.show()

    def plot_decision_boundary(model, axis):

        x0, x1 = np.meshgrid(
            np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
            np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
        )
        X_new = np.c_[x0.ravel(), x1.ravel()]

        y_predict = model.predict(X_new)
        zz = y_predict.reshape(x0.shape)

        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

        plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


