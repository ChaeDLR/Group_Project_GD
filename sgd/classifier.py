"""stochastic gradient descent (SGD) classifier example.
The SGD classifier is a linear model that can be used for classification tasks.
It is a stochastic optimization algorithm that minimizes the loss function by iteratively
updating the model parameters using a randomly selected subset of the training data.
The code trains the SGD classifier on the iris dataset and returns the accuracy score of
the predictions on the test set.
"""

import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def train_sgd_classifier(x_train, y_train):
    """Train a SGD classifier on the dataset and return the accuracy score."""

    # max number of iterations is 10^6 / number of samples in the training set
    max_iter = np.int64(np.ceil(10**6 / x_train.shape[0]))

    # create a pipeline that standardizes the data and then applies the SGD classifier
    clf = make_pipeline(
        StandardScaler(), linear_model.SGDClassifier(max_iter=max_iter, tol=1e-3)
    )

    # fit the model on the training data
    clf.fit(x_train, y_train)
    return clf
