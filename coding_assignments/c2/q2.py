# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt


def readMNISTdata():

    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows * ncols))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows * ncols))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

        # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate((np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate((np.ones([test_data.shape[0], 1]), test_data), axis=1)
    np.random.seed(314)
    np.random.shuffle(train_labels)
    np.random.seed(314)
    np.random.shuffle(train_data)

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels


def softmax(x):
    x_max = np.max(x, axis=1)[:, None]
    e_x = np.exp(x - x_max)
    for i in range(len(x)):
        e_x[i] /= np.sum(e_x[i], axis=0)
    return e_x


def cross_entropy(y_hat, y):
    # N = y.shape[0]
    y_log_pred = np.log(y_hat)
    loss = -np.sum(y * y_log_pred)

    return loss


def one_hot(y):
    # y_hot = np.zeros(shape=(y.shape[0], 10))
    target = y.reshape(-1)
    # y_hot[np.arange(len(y)), y.flatten()] = 1
    y_hot = np.eye(N_class)[target]
    return y_hot

# def l2_penalty(w):
#    return (w**2).sum() / 2
#


def predict(X, W, b, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here
    N = t.shape[0]
    theta = np.dot(X, W) + b
    y = softmax(theta)
    t_hat = one_hot(t)
    loss = cross_entropy(y, t_hat)
    pred = np.argmax(y, axis=1)

    acc = np.sum(pred == t.flatten()) / N
    return y, t_hat, loss, acc


def plot_train_graph(loss_list, epoch_list):
    plt.plot(epoch_list, loss_list)
    plt.title("The Learning Curve of the Training Cross-entropy Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("train_loss.png")
    plt.clf()


def plot_val_graph(acc_list, epoch_list):
    plt.plot(epoch_list, acc_list)
    plt.title("The Learning Curve of the Validation Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("Acc")
    plt.savefig("val_acc.png")
    plt.clf()


def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # TODO Your code here
    w = np.zeros((X_train.shape[1], N_class))
    b = np.zeros((N_class,))
    # w: (d+1)x1
    acc_val_list = []
    losses_train = []
    W_best = None
    b_best = None
    acc_best = 0.0
    epoch_best = 0
    epoch_list = [i for i in range(1, MaxEpoch + 1)]

    for epoch in range(MaxEpoch):
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train / batch_size))):
            X_batch = X_train[b * batch_size: (b + 1) * batch_size]
            y_batch = y_train[b * batch_size: (b + 1) * batch_size]

            y, y_hat_batch, loss_batch, _ = predict(X_batch, w, b, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            w = w - alpha * (1 / batch_size) * np.dot(X_batch.T, y - y_hat_batch)
            b = b - alpha * (1 / batch_size) * np.sum(y - y_hat_batch)

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoc
        # loss_this_epoch = loss_this_epoch / int(np.ceil(N_train / batch_size))
        loss_this_epoch = loss_this_epoch / N_train
        losses_train.append(loss_this_epoch)
        # print(loss_this_epoch)
        # 2.perform validation on the validation dataset
        _, _, _, acc_val = predict(X_val, w, b, t_val)
        acc_val_list.append(acc_val)
        # 3. keep track of the best validation epoch, acc, and weight
        current_acc_best = max(acc_best, acc_val)
        if current_acc_best != acc_best:
            acc_best = current_acc_best
            W_best = w
            b_best = b
            epoch_best = epoch
        print(
            "epoch:[{0}/{1}]\t"
            "train loss: {loss:.5f}\t"
            "acc:{acc:.5f}\t"
            "acc best:{acc_best:.5f}\t"
            "epoch best:{epoch_best}\t".format(epoch, MaxEpoch, loss=loss_this_epoch, acc=acc_val, acc_best=acc_best,
                                               epoch_best=epoch_best))

    plot_train_graph(losses_train, epoch_list)
    plot_val_graph(acc_val_list, epoch_list)
    return epoch_best, acc_best, W_best, b_best


##############################
# Main code starts here
alpha = 0.1  # learning rate
batch_size = 100  # batch size
MaxEpoch = 50  # Maximum epoch
decay = 0.01  # weight decay
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()

print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)

N_class = 10

epoch_best, acc_best, W_best, b_best = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, b_best, t_test)
# theta = np.dot(X_test, W_best) + b_best


print('At epoch', epoch_best, 'val: ', acc_best, 'test:', acc_test)
