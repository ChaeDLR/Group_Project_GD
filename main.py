import time

import matplotlib.pyplot as plt

import numpy as np

from math import trunc

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix

from sgd.classifier import train_sgd_classifier
from sgd.regressor import train_sgd_regressor
from batch_gd.train import train_batch_gd_regressor, train_batch_gd_classifier

from plot import hat_graph

def display_confusion_matrix(cm, class_names):
    fig, ax = plot_confusion_matrix(
        conf_mat=cm, class_names=class_names, figsize=(6, 6)
    )
    plt.title = "Confusion Matrix for SGD Classifier"
    plt.show()


def run_batch_gd_regressor(x_train, y_train, x_test):
    # Track time for GD
    start = time.time()

    # train the GD regressor on the training data
    batch_reg, scaler = train_batch_gd_regressor(x_train, y_train)

    batch_time = time.time() - start

    # Apply the same scaling used during training to the test data
    x_test_scaled = scaler.transform(x_test)

    # Use the trained Batch GD model to make predictions on the scaled test data
    y_pred_batch = batch_reg.predict(x_test_scaled)

    return y_pred_batch, batch_time


def run_batch_gd_classifier(x_train, y_train, x_test, y_test):
    start = time.time()

    batch_clf, scaler = train_batch_gd_classifier(x_train, y_train)

    batch_time = time.time() - start

    # Apply the same scaling used during training to the test data
    x_test_scaled = scaler.transform(x_test)

    y_pred_batch = batch_clf.predict(x_test_scaled)
    print(f"y_test={y_test}\n y_pred_batch={y_pred_batch}")

    # return the accuracy score of the predictions. 0.0 is the worst, 1.0 is the best
    print(
        f"Batch GD Classifier Accuracy: {metrics.accuracy_score(y_test, y_pred_batch)}"
    )

    # compute and display the confusion matrix for the test data
    cm = metrics.confusion_matrix(y_test, y_pred_batch)

    display_confusion_matrix(cm, dataset.target_names)


def run_sklearn_sgd_regressor(x_train, y_train, x_test):
    # Track time for SGD
    start = time.time()

    # train the SGD regressor on the training data
    sgd_reg = train_sgd_regressor(x_train, y_train)

    sgd_time = time.time() - start

    # predict the values of the test data
    y_pred = sgd_reg.predict(x_test)

    return y_pred, sgd_time


def run_sklearn_sgd_classifier(x_train, y_train, x_test, y_test):
    # train the SGD classifier on the training data
    sgd = train_sgd_classifier(x_train, y_train)

    # predict the classes of the test data
    y_pred = sgd.predict(x_test)

    # return the accuracy score of the predictions. 0.0 is the worst, 1.0 is the best
    print(f"SGD Classifier Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

    # compute and display the confusion matrix for the test data
    cm = metrics.confusion_matrix(y_test, y_pred)

    class_names = dataset.target_names

    display_confusion_matrix(cm, class_names)


def get_regressor_metrics(y_test, y_pred):
    # return the mean absolute error of the predictions. 0.0 is the best
    mae = metrics.mean_absolute_error(y_test, y_pred)

    # return the mean absolute error of the predictions. 0.0 is the best
    mse = metrics.mean_squared_error(y_test, y_pred)

    # return the root mean squared error of the predictions. 0.0 is the best
    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    return mae, mse, rmse

def truncate_metrics(metrics, decimals=2):
    factor = 10 ** decimals
    return [trunc(metric * factor) / factor for metric in metrics]

def print_metrics(name, time, mae, mse, rmse):
    print(f"{name} Performance:")
    print(f"    Training Time: {time:.6f} seconds")
    print(f"    Mean Absolute Error: {mae}")
    print(f"    Mean Squared Error: {mse}")
    print(f"    Root Mean Squared Error: {rmse}\n")

def plot_metrics(dataset_name, batch_metrics, sgd_metrics, save=False):
    fig, ax = plt.subplots(layout="constrained")

    x_labels = ["MAE", "MSE", "RMSE"]

    batch_metrics = truncate_metrics(batch_metrics)
    sgd_metrics = truncate_metrics(sgd_metrics)

    hat_graph(ax, x_labels, [batch_metrics, sgd_metrics], ["Batch GD", "SGD"])

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Error")
    ax.set_ylim(0, max(max(batch_metrics), max(sgd_metrics)) * 1.1)
    ax.set_title(f"Regressor Performance Comparison - {dataset_name}")
    ax.legend()

    if save:
        plt.savefig(f"./figures/{dataset_name}_regressor_comparison.png")
    else:
        plt.show()

if __name__ == "__main__":
    datasets_dict = {
        #"Wine": datasets.load_wine,    # classification dataset
        #"Iris": datasets.load_iris,    # classification dataset
        #"Linnerud": datasets.load_linnerud, # regression dataset
        "Diabetes": datasets.load_diabetes, # regression dataset
        }
    
    for name, loader in datasets_dict.items():

        dataset = loader()

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

        # train models
        y_pred_batch, batch_time = run_batch_gd_regressor(x_train, y_train, x_test)
        y_pred_sgd, sgd_time = run_sklearn_sgd_regressor(x_train, y_train, x_test)

        # get metrics, (mae, mse, rmse)
        batch_metrics = get_regressor_metrics(y_test, y_pred_batch)
        sgd_metrics = get_regressor_metrics(y_test, y_pred_sgd)

        # print metrics
        print_metrics("Batch GD Regressor", batch_time, *batch_metrics)
        print_metrics("SGD Regressor", sgd_time, *sgd_metrics)

        # plot metrics
        plot_metrics(f"{name}_dataset", batch_metrics, sgd_metrics, save=True)