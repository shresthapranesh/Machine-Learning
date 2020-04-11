import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# importing custom logistic regression model
from model import LogisticRegression

import os
from sklearn.metrics import accuracy_score

# random shuffling of dataset


def shuffling_files(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# split dataset in train, validation and test set at 8:1:1
def data_split(data):
    n_samples = len(data)
    itrain, ival, itest = int(
        0.8*n_samples), int(0.1*n_samples), int(0.1*n_samples)
    train, val, test = data[:itrain], data[itrain:itrain +
                                           ival], data[itrain+ival:]
    return train, val, test


#filepath = 'E:/Texas Tech University/Spring 2020/ECE-4332/Data/processed image/'
filepath = 'C:/Users/hp/Downloads/Output 2/'
# worm
worm_images = np.array([io.imread(filepath+'/Worm/'+image, as_gray=True)
                        for image in os.listdir(filepath+'/Worm/')])
worm_label = np.array([0 for i in worm_images])
# not a worm
noworm_images = np.array([io.imread(filepath+'/NoWorm/'+image, as_gray=True)
                          for image in os.listdir(filepath+'/NoWorm/')])
noworm_label = np.array([1 for i in noworm_images])

temp_X_train = np.concatenate((worm_images, noworm_images))
y_train = np.concatenate((worm_label, noworm_label))

X_data, y_data = shuffling_files(temp_X_train, y_train)

X_train, X_val, X_test = data_split(X_data)
y_train, y_val, y_test = data_split(y_data)

X_train = X_train/255
X_test = X_test/255


model = LogisticRegression(lr=0.01, epochs=500, lamb=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print('acc_test: {}'.format(accuracy_score(y_test, y_pred)))
