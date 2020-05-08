import numpy as np
from skimage import io
import os
from scipy.sparse import csr_matrix


def shuffling_files(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# split dataset in train, validation and test set at 8:1:1
def data_split(data):
    n_samples = len(data)
    itrain, itest = int(
        0.8*n_samples), int(0.2*n_samples)
    train, test = data[:itrain],  data[itrain:]
    return train, test


def create_dataset():
    filepath = '../Data/augmented/'
    # worm
    print('Loading worm images...')
    worm_images = np.array([io.imread(filepath+'/Worm/'+image, as_gray=True)
                            for image in os.listdir(filepath+'/Worm/')])
    worm_label = np.array([0 for i in worm_images])
    print('Done')
    # not a worm
    print('Loading noworm images...')
    noworm_images = np.array([io.imread(filepath+'/NoWorm/'+image, as_gray=True)
                              for image in os.listdir(filepath+'/NoWorm/')])
    noworm_label = np.array([1 for i in noworm_images])
    print('Done')

    temp_X_train = np.concatenate((worm_images, noworm_images))
    y_train = np.concatenate((worm_label, noworm_label))

    print('Shuffling images and labels ...')
    X_data, y_data = shuffling_files(temp_X_train, y_train)

    print('spliting data .....')
    X_train, X_test = data_split(X_data)
    y_train, y_test = data_split(y_data)
    print('Done')

    return X_train, y_train, X_test, y_test


def onehot_encoder(target):
    n_cases = target.shape[0]
    dummy = csr_matrix(
        (np.ones(n_cases), (np.arange(n_cases), target-target.min())))
    dummy = np.array(dummy.todense())
    return dummy


def accuracy_score(y_true, y_pred):
    x = np.zeros_like(y_true)
    x[y_true == y_pred] = 1
    return np.mean(x)
