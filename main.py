import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix

from sgd.classifier import train_sgd_classifier


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
    sgd = train_sgd_classifier(x_train, y_train)

    # predict the classes of the test data
    y_pred = sgd.predict(x_test)

    # return the accuracy score of the predictions. 0.0 is the worst, 1.0 is the best
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

    cm = metrics.confusion_matrix(y_test, y_pred)

    class_names = dataset.target_names

    display_confusion_matrix(cm, class_names)