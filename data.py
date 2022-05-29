import numpy as np
import os


def load_data(train=True):
    """
    Load the data from disk

    Parameters
    ----------
    train : bool
        Load training data if true, else load test data
    Returns
    -------
        Tuple:
            Images
            Labels
    """
    directory = 'train' if train else 'test'
    patterns = np.load(os.path.join(
        './data/', directory, 'images.npz'))['arr_0']
    labels = np.load(os.path.join('./data/', directory, 'labels.npz'))['arr_0']
    return patterns.reshape(len(patterns), -1), labels


def z_score_normalize(X, u=None, xd=None):
    """
    Performs z-score normalization on X.

    f(x) = (x - μ) / σ
        where
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    if not u:
        u = np.mean(X)

    if not xd:
        xd = np.std(X)

    return (X-u)/xd


def min_max_normalize(X, _min=None, _max=None):
    """
    Performs min-max normalization on X.

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize
    _min (optional) : np.array
        The min to use when normalizing
    _max (optional) : np.array
        The max to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """
    if not _min:
        _min = np.min(X)
    if not _max:
        _max = np.max(X)

    return (X-_min)/(_max-_min)


def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    # find maximum element, add 1 to get # of columns to create
    n_max = np.max(y)+1

    # create identity matrix with n_max columns, for each arg in y, create a new row and set index arg to 1
    return np.eye(n_max)[y]  # sheesh
    # n = len(np.unique(y))
    # return np.eye(n)


def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    # return index of maximum element in each row
    return np.argmax(y, axis=1)


def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together.
    Ideas:
        NumPy array indexing
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    # generate a permutation of indices from 0 to # of imgs
    shuffled_indices = np.random.permutation(len(dataset[0]))

    # use permutations to index data
    output_X = np.array(dataset[0])[shuffled_indices]
    output_y = np.array(dataset[1])[shuffled_indices]

    return (output_X, output_y)


def append_bias(X):
    """
    Append bias term for dataset.

    Parameters
    ----------
    X
        2d numpy array with shape (N,d)

    Returns
    -------
        2d numpy array with shape ((N+1),d) --> edit: should be ((N), d+1)
    """

    # create a 2d array of 1's (aka another column with all 1's), then append it to the original tensor
    # np.ones((#rows, #cols))
    return np.append(X, np.ones((len(X), 1)), axis=1)


def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def generate_k_fold_set(dataset, k=10):
    X, y = dataset

    order = np.random.permutation(len(X))

    fold_width = len(X) // k

    l_idx, r_idx = 0, fold_width

    for i in range(k):
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate(
            [y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
        yield train, validation
        l_idx, r_idx = r_idx, r_idx + fold_width
