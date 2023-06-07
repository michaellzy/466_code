import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def data_preprocess():
    data = pd.read_csv("./spambase/spambase.csv")
    target_arr = []
    data_target = data["class"]

    for ele in data_target.items():
        target_arr.append(ele[1])
    y_ = np.array(target_arr)

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], y_, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0)

    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

    X_train = np.concatenate((np.ones([x_train.shape[0], 1]), x_train), axis=1)
    X_test = np.concatenate((np.ones([x_test.shape[0], 1]), x_test), axis=1)

    y_train = y_train.reshape([-1, 1])
    y_test = y_test.reshape([-1, 1])

    return X_train, y_train, X_test, y_test


def calc_accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


def majority_guess(y_train, y_test):
    prediction = np.bincount(y_train.flatten()).argmax()
    guess = np.array([prediction for i in range(len(y_test))])
    guess = guess.reshape([-1, 1])
    return guess


def decision_tree(X_train, y_train, X_test):
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred = y_pred.reshape([-1, 1])
    return y_pred


def knn(X_train, y_train, X_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_pred = y_pred.reshape([-1, 1])
    return y_pred


def neural_network(X_train, y_train, X_test, opt, max_iter):
    nn = MLPClassifier(solver=opt, activation='relu', alpha=1e-3, hidden_layer_sizes=(5, 2), max_iter=max_iter)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    y_pred = y_pred.reshape([-1, 1])
    return y_pred


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = data_preprocess()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    maj_guess = majority_guess(y_train, y_test)
    acc_maj = calc_accuracy(maj_guess, y_test)
    print("Acc of majority guess: %.5f" % acc_maj)

    dt = decision_tree(X_train, y_train.flatten(), X_test)
    acc_nb = calc_accuracy(dt, y_test)
    print("Acc of decision tree: %.5f" % acc_nb)

    for k in range(3, 6):
        knn_res = knn(X_train, y_train.flatten(), X_test, k)
        acc_knn = calc_accuracy(knn_res, y_test)
        print("Acc of knn: %.5f with k = %d" % (acc_knn, k))

    for opt in ["adam", "sgd"]:
        for max_iter in [100, 200]:
            nn = neural_network(X_train, y_train.flatten(), X_test, opt=opt, max_iter=max_iter)
            acc_nn = calc_accuracy(nn, y_test)
            print("iteration:[%d], opt:[%s], Acc of neural network: %.5f" % (max_iter, opt, acc_nn))
