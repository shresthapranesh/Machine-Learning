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
    filepath = '../Data/processed image/'
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


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding, stride=1):
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)
    cols = x[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):

    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def onehot_encoder(target):
    n_cases = target.shape[0]
    dummy = csr_matrix(
        (np.ones(n_cases), (np.arange(n_cases), target-target.min())))
    dummy = np.array(dummy.todense())
    return dummy
