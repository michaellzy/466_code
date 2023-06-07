from utils import plot_data, generate_data
import numpy as np

"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""


def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    # t = t.reshape((n_train, 1))
    alpha = 0.1
    w = np.zeros((X.shape[1], ))
    # b = np.zeros((X.shape[0], ))
    b = 0
    for epoch in range(50):
        y_hat = predict_logistic_regression(X, w, b)
        w = w - alpha * np.dot(X.T, y_hat - t)
        b = b - alpha * (y_hat - t)

    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    z = np.dot(X, w) + b
    t = np.round(1 / (1 + np.exp(-z)))
    return t


def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, t))
    b = np.zeros(shape=(400, ))

    return w,b


def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    t = np.dot(X, w) + b
    return np.round(t)


def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    target = t.flatten()
    n = target.shape[0]
    acc = (target == t_hat).sum() / n
    return acc


def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    # w = w.flatten()
    # b = b.flatten()
    plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)

    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_B_logistic.png')


main()
