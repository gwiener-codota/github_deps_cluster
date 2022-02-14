import numpy as np


def pair_share(x, y):
    intersect = x.dot(y.transpose()).sum()
    union = ((x + y) > 0).sum()
    return intersect / union if union != 0 else 0


def _main():
    from sklearn.cluster import DBSCAN
    X = [
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 1]
    ]
    scan = DBSCAN(eps=0.5, min_samples=1, metric=pair_share, leaf_size=1)
    y = scan.fit_predict(X)
    print(y)


if __name__ == '__main__':
    _main()
