import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from model import LogisticRegression

import time

X_train = idx2numpy.convert_from_file(
    'E:/Texas Tech University/Spring 2020/ECE-4332/Data/train-images.idx3-ubyte')
y_train = idx2numpy.convert_from_file(
    'E:/Texas Tech University/Spring 2020/ECE-4332/Data/train-labels.idx1-ubyte')

X_test = idx2numpy.convert_from_file(
    'E:/Texas Tech University/Spring 2020/ECE-4332/Data/t10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file(
    'E:/Texas Tech University/Spring 2020/ECE-4332/Data/t10k-labels.idx1-ubyte')

# batch normalization
X_train = X_train/255
X_test = X_test/255

model = LogisticRegression(lr=0.34)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print('acc_test: {}'.format(model.accuracy_score(y_test, y_pred)))
