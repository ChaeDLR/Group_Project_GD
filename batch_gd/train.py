from sklearn.preprocessing import StandardScaler

from .regressor import BatchGDRegressor


def train_batch_gd_regressor(x_train, y_train):

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)

    reg = BatchGDRegressor(
        learning_rate=0.01,
        epochs=1000
    )

    reg.fit(x_train_scaled, y_train)

    return reg, scaler