import argparse

from numpy import Inf
import numpy as np
import network
import data
import image
from copy import deepcopy
from matplotlib import pyplot as plt


# filter to two classes, normalize, and one hot encode labels
def transform_dataset(x, y, imgs, labels):
    imgs_xy, labels_xy = [], []
    for d in range(len(labels)):
        if labels.item(d) == x or labels.item(d) == y:
            imgs_xy.append(imgs[d])
            labels_xy.append(labels[d])

    imgs_xy = data.z_score_normalize(imgs_xy)
    labels_xy = data.onehot_encode(labels_xy)
    return np.array(imgs_xy), np.array(labels_xy)

# Imports the training and test data and shuffles them before we start using
def import_and_shuffle():
    # import data
    imgs_train, labels_train = data.load_data(train=True)
    imgs_test, labels_test = data.load_data(train=False)

    # shuffle the data
    imgs_train, labels_train = data.shuffle((imgs_train, labels_train))
    imgs_test, labels_test = data.shuffle((imgs_test, labels_test))

    return imgs_train, labels_train, imgs_test, labels_test

# Function for performing cross validation
def cross_validation(imgs_train, labels_train, hyperparameters, binary=False):
    """
    Parameters
    ----------
        imgs_train : images to train on
        labels_train: labels to train on
        hyperparameters: contains batch size, # of epochs, learning rate, normalization method, and # of folds
        binary : whether to do cross validation for binary or multiclass classifier

    Returns
    -------
        best_val_accs : best validation accuracies from each fold
        fold_train_losses : average training losses from each fold
        fold_val_losses : average validation losses from each fold
        train_losses : training losses from each batch
        val_losses : validation losses from each batch
    """

    # list of networks
    fold_networks = []
    val_losses, val_accs = [], []
    train_losses = []

    # Losses per fold
    fold_val_losses = []
    fold_train_losses = []

    best_val_accs = []
    fold_num = 1
    break_test = False

    # Loop over 10 folds
    for train_set, validation_set in data.generate_k_fold_set((imgs_train, labels_train)):
        print("fold #%i..." % fold_num)
        fold_num += 1
        highest_epoch = 0

        # add the newest network for our new fold
        if binary:
            fold_networks.append(network.Network(hyperparameters,
                                                 network.sigmoid, network.binary_cross_entropy, out_dim=1))
        else:
            # make multiclass network
            fold_networks.append(network.Network(hyperparameters,
                                                 network.softmax, network.multiclass_cross_entropy, out_dim=10))

        # Loop over epochs
        for _ in range(hyperparameters.epochs):
            train_set = data.shuffle(train_set)

            # generate minibatches
            minibatches = data.generate_minibatches(
                train_set, hyperparameters.batch_size)

            # train on all of the minibatches
            curr_network = fold_networks[-1]
            for mb in minibatches:
                if binary:
                    curr_network.train(mb, binary=True)
                else:
                    curr_network.train(mb, binary=False)

            if break_test:  # if needed, break early for debugging purposes
                break

            if binary:
                # calculate loss and accuracy on validation set
                val_loss, val_acc = curr_network.test(
                    validation_set, binary=True)
                val_losses.append(val_loss)

                # calculate loss on train set
                train_loss = curr_network.test(train_set, binary=True)[0]
                train_losses.append(train_loss)

            # multiclass case
            else:
                # calculate loss and accuracy on validation set
                val_loss, val_acc = curr_network.test(
                    validation_set, binary=False)
                val_losses.append(val_loss)

                # calculate loss on train set
                train_loss = curr_network.test(train_set, binary=False)[0]
                train_losses.append(train_loss)

            val_accs.append(val_acc)

            # Early stop if current epoch was worse than the last one
            if len(val_accs) > 1 and (val_accs[-1] < val_accs[-2]):
                val_accs.pop()
                # record best validation accuracy
                best_val_accs.append(val_accs[-1])
                val_accs = []
                break
            highest_epoch += 1

        print("\t highest epoch = %d \n" % highest_epoch)

        if break_test:  # if needed, break early for debugging purposes
            break

        # Get average train and validation loss for one fold
        fold_train_losses.append(np.average(np.array(train_losses)))
        fold_val_losses.append(np.average(np.array(val_losses)))

    return best_val_accs, fold_train_losses, fold_val_losses, train_losses, val_losses


def main(hyperparameters):
    # import and shuffle data
    imgs_train, labels_train, imgs_test, labels_test = import_and_shuffle()

    # choose:
    #   1 for class 0 vs class 6
    #   2 for class 2 vs class 6
    #   3 for all classes
    choose = -1
    valid_choose = [1, 2, 3]
    while int(choose) not in valid_choose:
        print("Enter 1 for class 0 and class 6 binary classification")
        print("Enter 2 for class 2 and class 6 binary classification")
        print("Enter 3 for all class multiclass classification")
        choose = input("Enter: ")
    print()
    choose = int(choose)

    # CLASS 0 VS CLASS 6
    if choose == 1:
        # create train class 0/6 dataset
        imgs_train_06, labels_train_06 = transform_dataset(
            0, 6, imgs_train, labels_train)

        # create test class 0/6 dataset
        imgs_test_06, labels_test_06 = transform_dataset(
            0, 6, imgs_test, labels_test)

        print("imgs_test_06 shape = ", imgs_test_06.shape)
        print("labels_test_06 shape = ", labels_test_06.shape)
        
        # perform cross validation
        best_val_accs, fold_train_losses, fold_val_losses, train_losses, val_losses = cross_validation(
            imgs_train_06, labels_train_06, hyperparameters, binary=True)

        # check validation accuracies
        print("best validation accs = ", best_val_accs)
        print("avg validation acc = ", np.average(np.array(best_val_accs)))
        print()

        # create a brand new network to train on the entire training set now
        net_06 = network.Network(
            hyperparameters, network.sigmoid, network.binary_cross_entropy, out_dim=1)

        print("Training net06...")

        minibatches = data.generate_minibatches((imgs_train_06, labels_train_06),
                                                hyperparameters.batch_size)

        for mb in minibatches:
            net_06.train(mb, binary=True)

        # Get model output
        net06_test_loss, net06_test_acc = net_06.test(
            (imgs_test_06, labels_test_06), binary=True)

        print("net06 test loss = ", net06_test_loss)
        print("net06 test acc = ", net06_test_acc)

        # PLOTTING ___________________________________
        # plt.plot(fold_train_losses)
        # plt.plot(fold_val_losses)
        plt.plot(train_losses, color='blue', label="training loss")
        plt.plot(val_losses, color='red', label="validation loss")
        plt.legend()
        plt.xlabel("# of epochs")
        plt.ylabel("loss")
        plt.show(block=True)
        # ____________________________________

    # CLASS 2 VS CLASS 6
    elif choose == 2:
        # create train class 2/6 dataset
        imgs_train_26, labels_train_26 = transform_dataset(
            2, 6, imgs_train, labels_train)

        # create test class 2/6 dataset
        imgs_test_26, labels_test_26 = transform_dataset(
            2, 6, imgs_test, labels_test)

        # perform cross validation
        best_val_accs, fold_train_losses, fold_val_losses, train_losses, val_losses = cross_validation(
            imgs_train_26, labels_train_26, hyperparameters, binary=True)

        print("best validation accs = ", best_val_accs)
        print("avg validation acc = ", np.average(np.array(best_val_accs)))
        print()

        # create brand new network to work with entire training set
        net_26 = network.Network(
            hyperparameters, network.sigmoid, network.binary_cross_entropy, out_dim=1)

        print("Training net26...")
        minibatches = data.generate_minibatches((imgs_train_26, labels_train_26),
                                                hyperparameters.batch_size)

        for mb in minibatches:
            net_26.train(mb, binary=True)

        # # don't use minibatches for this one because of overfitting
        # net_26.train((imgs_test_26, labels_test_26), binary=True)

        # get model output
        net26_test_loss, net26_test_acc = net_26.test(
            (imgs_test_26, labels_test_26), binary=True)

        print("net26 test loss = ", net26_test_loss)
        print("net26 test acc = ", net26_test_acc)

        # PLOTTING ___________________________________
        # plt.plot(fold_train_losses)
        # plt.plot(fold_val_losses)
        plt.plot(train_losses, color='blue', label="training loss")
        plt.plot(val_losses, color='red', label="validation loss")
        plt.legend()
        plt.xlabel("# of epochs")
        plt.ylabel("loss")
        plt.show(block=True)
        # ____________________________________

    # ALL CLASSES
    else:
        # Z-Score Normalize the whole training set
        # one-hot encode all the training labels
        imgs_train_all, labels_train_all = data.z_score_normalize(
            imgs_train), data.onehot_encode(labels_train)
        # Same thing for the whole test set
        imgs_test_all, labels_test_all = data.z_score_normalize(
            imgs_test), data.onehot_encode(labels_test)

        # perform cross validation
        best_val_accs, fold_train_losses, fold_val_losses, train_losses, val_losses = cross_validation(
            imgs_train_all, labels_train_all, hyperparameters, binary=False)

        # create new network
        multiclass_net = network.Network(
            hyperparameters, network.softmax, network.multiclass_cross_entropy, out_dim=10)

        print("best validation accs = ", best_val_accs)
        print("avg validation acc = ", np.average(np.array(best_val_accs)))
        print()

        # do SGD (without epoches)
        minibatches = data.generate_minibatches((imgs_train_all, labels_train_all),
                                                hyperparameters.batch_size)

        for mb in minibatches:
            multiclass_net.train(mb, binary=False)

        multiclass_loss, multiclass_acc = multiclass_net.test(
            (imgs_test_all, labels_test_all), binary=False)

        print("multiclass test loss = ", multiclass_loss)
        print("multiclass test acc = ", multiclass_acc)
        print()

        '''
        # For loop to determine the best hyperparameters for the best average validation accuracy
        

        # output array for average validation accuracies
        outputs = []

        # for each batch size in our range:
        for bat in range(1, 2):
            # for each learning rate in our range:
            # for lr in range(0.0001, 0.0005, 0.0001):
            for lr in np.linspace(0.0002, 0.0007, 30):
                # update the hyperparameters
                hyperparameters.batch_size = bat
                hyperparameters.learning_rate = lr

                # create a new model
                multiclass_net = None
                multiclass_net = network.Network(
                    hyperparameters, network.softmax, network.multiclass_cross_entropy, out_dim=10)

                # generate new minibatch data
                minibatches = data.generate_minibatches((imgs_train_all, labels_train_all),
                                                        hyperparameters.batch_size)
                for mb in minibatches:
                    multiclass_net.train(mb, binary=False)
                best_val_accs, fold_train_losses, fold_val_losses, train_losses, val_losses = cross_validation(
                    imgs_train_all, labels_train_all, hyperparameters, binary=False)
                outputs.append((best_val_accs, bat, lr))
                print("finished loop: ", hyperparameters.batch_size,
                      hyperparameters.learning_rate)

        outputs.sort(reverse=True)

        print("multiclass test loss = ", outputs[0][1])
        print("multiclass test acc = ", outputs[0][0])
        print(outputs[0])

        '''

        # PLOTTING ___________________________________
        # plt.plot(fold_train_losses)
        # plt.plot(fold_val_losses)
        plt.plot(train_losses, color='blue', label="training loss")
        plt.plot(val_losses, color='red', label="validation loss")
        plt.legend()
        plt.xlabel("# of epochs")
        plt.ylabel("loss")
        plt.show(block=True)
        # ____________________________________

        # Heatmap Plotting for Softmax ______________________
        # multi_weights = multiclass_net.get_weights()
        # plt.figure(0)
        # counter = 0
        # for i in range(5):
        #     for j in range(2):
        #         ax = plt.subplot2grid((5,2), (i,j))
        #         ax.imshow(multi_weights[:,counter][:-1].reshape((28,28)), cmap='hot', interpolation='nearest')
        #         counter += 1
        # plt.show()
        # ____________________________________


parser = argparse.ArgumentParser(description='CSE151B PA1')
parser.add_argument('--batch-size', type=int, default=1,
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--learning-rate', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--z-score', dest='normalization', action='store_const',
                    default=data.min_max_normalize, const=data.z_score_normalize,
                    help='use z-score normalization on the dataset, default is min-max normalization')
parser.add_argument('--k-folds', type=int, default=10,
                    help='number of folds for cross-validation')

hyperparameters = parser.parse_args()
print()
print("batch size = ", hyperparameters.batch_size)
print("epochs = ", hyperparameters.epochs)
print("learning rate = ", hyperparameters.learning_rate)
print("k-folds = ", hyperparameters.k_folds)
print()
main(hyperparameters)
