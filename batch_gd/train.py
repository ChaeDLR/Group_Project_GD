from numpy import ndarray

from sklearn.preprocessing import StandardScaler

from .regressor import BatchGDRegressor
from .classifier import BatchGDClassifier


def train_batch_gd_regressor(x_train: ndarray, y_train: ndarray, learning_rate=0.01, epochs=1000):

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)

    reg = BatchGDRegressor(
        learning_rate=learning_rate,
        epochs=epochs
    )

    reg.fit(x_train_scaled, y_train)

    return reg, scaler

def train_batch_gd_classifier(x_train: ndarray, y_train: ndarray, learning_rate=0.01, epochs=1000):

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)

    clf = BatchGDClassifier(
        learning_rate=learning_rate,
        epochs=epochs
    )

    clf.fit(x_train_scaled, y_train)

    return clf, scaler