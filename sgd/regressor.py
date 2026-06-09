import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def train_sgd_regressor(x_train, y_train):
    max_iter = np.int64(np.ceil(10**6 / x_train.shape[0]))

    reg = make_pipeline(
        StandardScaler(), linear_model.SGDRegressor(max_iter=max_iter, tol=1e-3)
    )
    reg.fit(x_train, y_train)
    return reg
