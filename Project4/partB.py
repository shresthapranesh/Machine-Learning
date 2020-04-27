import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# importing custom logistic regression model
from model import LogisticRegression

import os
from sklearn.metrics import accuracy_score
import time
# random shuffling of dataset


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


#filepath = 'E:/Texas Tech University/Spring 2020/ECE-4332/Data/processed image/'
filepath = 'C:/Users/hp/Downloads/Output 2/'
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

X_train = X_train/255
X_test = X_test/255


model = LogisticRegression(lr=0.02, epochs=500, lamb=8)
tic1 = time.time()
model.fit(X_train, y_train)
toc1 = time.time()

tic2 = time.time()
y_pred = model.predict(X_test)
toc2 = time.time()
y_pred = np.argmax(y_pred, axis=1)

print('Training Time: {}'.format(toc1-tic1))
print('Testing Time: {}'.format(toc2-tic2))
print('acc_test: {}'.format(accuracy_score(y_test, y_pred)))
plt.show()
