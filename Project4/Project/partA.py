import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mlxtend.data import loadlocal_mnist
from sklearn import metrics

import time


class LogisticRegression_():
    def __init__(self, X_train, train_labels, X_test, test_labels):
        self.X_train = X_train
        self.train_labels = train_labels
        self.X_test = X_test
        self.test_labels = test_labels

    def softmax(self, x, w):

        z = x@w
        z -= np.max(z, axis=1).reshape((-1, 1))
        sm = np.exp(z) / np.exp(z).sum(axis=1).reshape((-1, 1))
        return sm

    def costFunction(self, N, smax, y_enc, lam, X_train, W):
        # loss function for softmax regression + regularization factor
        cost = (-1 / N) * (np.sum(y_enc * np.log(smax)) -
                           (lam / 2) * np.sum(W * W))
        diff = (smax - y_enc)
        grad = (X_train.T@diff) + (lam / 2) * W
        return cost, grad

    def getProbPred(self, X_data, w):
        probability = self.softmax(X_data, w)
        prediction = np.argmax(probability, axis=1)
        return probability, prediction

    def getAccuracy(self, predictors, target, weight):
        prob, prede = self.getProbPred(predictors, weight)
        sum = 0
        for ite in range(len(target)):
            if prede[ite] == target[ite]:
                sum += 1
        return sum / len(target)

    def oneHotEncoding(self, labels, numClass):
        y_enc = []
        for num in labels:
            row = np.zeros(numClass)
            row[num] = 1
            y_enc.append(row)
        y_enc = np.array(y_enc)
        return y_enc

    def designMatrix(self, mat):
        mat = mat / np.max(mat)
        ones = np.ones((mat.shape[0], 1))
        mat = np.hstack((mat, ones))
        examples, features = mat.shape
        return mat, examples, features

    def train(self):
        t0 = time.time()

        X, N, features = self.designMatrix(self.X_train)
        # w = np.zeros((features, len(np.unique(self.train_labels))))
        classK = len(np.unique(self.train_labels))
        w = np.random.uniform(
            0, 1, (features, classK))
        y_enc = self.oneHotEncoding(self.train_labels, classK)

        learning_rate = 1e-5
        lamda = 2
        loss_matrix = []
        acc_matrix = []
        stopping_criteria = 100
        J_cost = 0
        ite = 0
        while stopping_criteria >= 0.0001:
            # while ite != 10:
            ite += 1
            smax = self.softmax(X, w)
            prev_J = J_cost
            J_cost, grad = self.costFunction(N, smax, y_enc, lamda, X, w)
            w = w - learning_rate * grad
            stopping_criteria = abs(prev_J - J_cost)
            loss_matrix.append(J_cost)

            acc = self.getAccuracy(X, self.train_labels, w)
            acc_matrix.append(acc)
            print(
                f"Loss: {round(J_cost,4):.4f},    Train Accuracy: {round(acc,4):.4f}")

        t1 = time.time()
        train_time = abs(t1 - t0)

        print(f"Number of Iterations: {ite}")
        print(f"Cost: {round(J_cost,4):.4f}")

        X_test, _, _a = self.designMatrix(self.X_test)
        t0 = time.time()
        testAccuracy = self.getAccuracy(X_test, self.test_labels, w)
        t1 = time.time()
        test_time = abs(t1 - t0)
        print(f"Testing Accuracy: {testAccuracy}")

        print(f"Training Time: {round(train_time,4):.4f} s")
        print(f"Testing Time: {round(test_time,4):.4f} s")

        plt.figure(dpi=100)
        plt.plot(loss_matrix)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.title('Cost Function for Softmax Linear Regression')

        plt.figure(dpi=100)
        plt.plot(acc_matrix)
        plt.ylabel('Accuracy')
        plt.xlabel('Iterations')
        plt.title('Accuracy for Softmax Linear Regression')

        prob, pred = self.getProbPred(X, w)
        cm = metrics.confusion_matrix(self.train_labels, pred)
        plt.figure(dpi=100)
        sns.heatmap(cm, annot=True, fmt=".1f", linewidths=0.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title(
            'Confusion Matrx for Softmax Linear Regression For MNIST DataSet with Linear Model')


if __name__ == "__main__":
    print("Loading Dataset...")

    X_train, train_labels = loadlocal_mnist(
        images_path='MNIST/train-images.idx3-ubyte',
        labels_path='MNIST/train-labels.idx1-ubyte')
    X_test, test_labels = loadlocal_mnist(
        images_path='MNIST/t10k-images.idx3-ubyte',
        labels_path='MNIST/t10k-labels.idx1-ubyte')

    newmodel = LogisticRegression_(X_train, train_labels, X_test, test_labels)
    print("Training Model...")
    newmodel.train()

    plt.tight_layout()
    plt.show()
