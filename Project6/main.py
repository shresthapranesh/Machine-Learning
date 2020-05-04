from dense_layer import FullyConnected
from conv_layer import Conv2D
from maxpool import Maxpool2D
from utils import create_dataset, onehot_encoder
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import io
import idx2numpy
from sklearn.metrics import accuracy_score

learning_rate = 1e-1

# Model
conv = Conv2D(8, (3, 3), lr=learning_rate)
pool = Maxpool2D(pool=(2, 2), stride=(2, 2))
dense = FullyConnected(layers=[32, 10], lr=learning_rate)


def forward(image):
    out1 = conv.forward_pass(image)
    out2 = pool.forward_pass(out1)
    out3 = dense.forward_pass(out2)
    return out3


def backprop(label):
    gradient1 = dense.backprop(label)
    gradient2 = pool.backprop(gradient1)
    gradient3 = conv.backprop(gradient2)


def train(image, label):
    output = forward(image)
    backprop(label)
    loss = -np.sum(label*np.log(output))*1/output.shape[1]
    print('Loss: {}'.format(loss))


train_data = np.load('train_data.npz')
test_data = np.load('test_data.npz')

X_train, y_train = train_data['X_train'], train_data['y_train']
X_test, y_test = test_data['X_test'], test_data['y_test']


# X_train = idx2numpy.convert_from_file('../Data/train-images.idx3-ubyte')
# y_train = idx2numpy.convert_from_file('../Data/train-labels.idx1-ubyte')

# X_test = idx2numpy.convert_from_file('../Data/t10k-images.idx3-ubyte')
# y_test = idx2numpy.convert_from_file('../Data/t10k-labels.idx1-ubyte')


X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

y_train = onehot_encoder(y_train).T
#y_test = onehot_encoder(y_test).T

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


# for i in range(3):
#     train(X_train/255, y_train)

# temp = conv.forward_pass(X_test/255)
# temp = pool.forward_pass(temp)
# y_pred = dense.forward_pass(temp)

# y_pred = np.argmax(y_pred, axis=0)
# acc_score = accuracy_score(y_test, y_pred)
# print('accuracy Score: {}'.format(acc_score))

# plt.figure()
# plt.imshow(image[0, :, :], cmap=plt.cm.gray)


# plt.figure()
# for i in range(8):
#     plt.subplot(2, 4, i+1)
#     plt.imshow(conv.filters[i, 0, :, :], cmap=plt.cm.gray)

# plt.show()
