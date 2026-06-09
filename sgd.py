"""stochastic gradient descent (SGD) classifier example using the iris dataset.
The SGD classifier is a linear model that can be used for classification tasks.
It is a stochastic optimization algorithm that minimizes the loss function by iteratively
updating the model parameters using a randomly selected subset of the training data.
The code trains the SGD classifier on the iris dataset and returns the accuracy score of
the predictions on the test set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model, metrics, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix


def test_sgd():
    x_train = np.array([[0, 0], [1, 1], [1, 2], [2, 3]])
    y_train = np.array([1, 1, 2, 2])
    x_test = np.array([[2, 2]])
    clf = make_pipeline(
        StandardScaler(), linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    )
    clf.fit(x_train, y_train)
    return clf.predict(x_test)


def train_sgd(x_train, y_train):
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


def display_confusion_matrix(cm, class_names):
    fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=class_names, figsize=(6, 6))
    plt.title=("Confusion Matrix for SGD Classifier on Wine Dataset")
    plt.show()


if __name__ == "__main__":

    # can change the dataset here
    dataset = datasets.load_wine()
    #dataset = datasets.load_iris()

    # features, 2d matrix of floats
    # each row is a sample, each column is a feature
    x = dataset.data

    # labels, 1d array of ints
    # each element is the class label for the corresponding row in x
    y = dataset.target

    # split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    # train the SGD classifier on the training data
    sgd = train_sgd(x_train, y_train)

    # predict the classes of the test data
    y_pred = sgd.predict(x_test)

    # return the accuracy score of the predictions. 0.0 is the worst, 1.0 is the best
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

    cm = metrics.confusion_matrix(y_test, y_pred)

    class_names = dataset.target_names

    display_confusion_matrix(cm, class_names)
