import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# acknowledge: skeleton code from coding assignment 2


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def predict(X, w, y=None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    z = np.dot(X, w)
    y_hat = sigmoid(z)
    loss = calc_loss_function(y_hat, y)
    pred = np.round(y_hat)
    acc = calc_accuracy(pred, y)
    return y_hat, loss, acc


def calc_accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


def calc_loss_function(y_hat, t):
    N = t.shape[0]
    cost1 = -t * np.log(y_hat)
    cost2 = (1 - t) * np.log(1 - y_hat)
    loss = cost1 - cost2
    loss = loss.sum() / N
    return loss


def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    w = np.zeros((X_train.shape[1], N_class))

    # w: (d+1)x1
    acc_val_list = []
    losses_train = []
    W_best = None
    acc_best = 0.0
    epoch_best = 0
    epoch_list = [i for i in range(1, MaxEpoch + 1)]

    for epoch in range(MaxEpoch):
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train / batch_size))):
            X_batch = X_train[b * batch_size: (b + 1) * batch_size]
            y_batch = y_train[b * batch_size: (b + 1) * batch_size]

            y_hat_batch, loss_batch, acc = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # Mini-batch gradient descent
            w = w - alpha * (1 / batch_size) * np.dot(X_batch.T, y_hat_batch - y_batch)

        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoc
        # loss_this_epoch = loss_this_epoch / int(np.ceil(N_train / batch_size))
        loss_this_epoch = loss_this_epoch / N_train
        losses_train.append(loss_this_epoch)
        # print(loss_this_epoch)
        # 2.perform validation on the validation dataset
        _, _, acc_val = predict(X_val, w, t_val)
        acc_val_list.append(acc_val)
        # 3. keep track of the best validation epoch, acc, and weight
        current_acc_best = max(acc_best, acc_val)
        if current_acc_best != acc_best:
            acc_best = current_acc_best
            W_best = w
            epoch_best = epoch
        print(
            "epoch:[{0}/{1}]\t"
            "alpha:{2}\t"
            "train loss: {loss:.5f}\t"
            "acc:{acc:.5f}\t"
            "acc best:{acc_best:.5f}\t"
            "epoch best:{epoch_best}\t".format(epoch, MaxEpoch, alpha, loss=loss_this_epoch, acc=acc_val,
                                               acc_best=acc_best,
                                               epoch_best=epoch_best))

    # plot_train_graph(losses_train, epoch_list)
    # plot_val_graph(acc_val_list, epoch_list)
    return epoch_best, acc_best, W_best


if __name__ == '__main__':
    # alpha = 0.1  # learning rate
    batch_size = 100  # batch size
    # MaxEpoch = 50  # Maximum epoch
    alpha_list = [0.1, 0.01]
    MaxEpoch_list = [20, 50]
    N_class = 1

    data = pd.read_csv("./spambase/spambase.csv")
    target_arr = []
    data_target = data["class"]

    for ele in data_target.items():
        target_arr.append(ele[1])
    y_ = np.array(target_arr)

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], y_, test_size=0.2, random_state=0)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0)

    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    x_val = (x_val - np.mean(x_val, axis=0)) / np.std(x_val, axis=0)
    x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

    X_train = np.concatenate((np.ones([x_train.shape[0], 1]), x_train), axis=1)
    X_val = np.concatenate((np.ones([x_val.shape[0], 1]), x_val), axis=1)
    X_test = np.concatenate((np.ones([x_test.shape[0], 1]), x_test), axis=1)

    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)

    y_train = y_train.reshape([-1, 1])
    y_val = y_val.reshape([-1, 1])
    y_test = y_test.reshape([-1, 1])

    for alpha in alpha_list:
        for MaxEpoch in MaxEpoch_list:
            _, _, w = train(X_train, y_train, X_val, y_val)
            y_hat_test, _, acc_test = predict(X_test, w, y_test)
            print("test accuracy : %.5f" % acc_test)
            print("---------------------------------------------")
