import numpy as np
from scipy.sparse import csr_matrix


class LogisticRegression:
    def __init__(self, lr=0.001, tolerance=1e-10, nclass=None, degree=2, epochs=300, lamb=4):
        self.lr = lr
        self.tolerance = tolerance
        self.nclass = nclass
        self.degree = degree
        self.epochs = epochs
        self.lamb = lamb

    def fit(self, X, target):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))
            n_data = X.shape[0]
            n_features = X.shape[1]
        else:
            raise Exception("Input is in Incorrect Format!!")

        self.weight = np.random.rand(n_features, len(np.unique(target)))

        t = self.dummy_gen(target)

        for i in range(self.epochs):

            y = self.softmax(X, self.weight)

            # gradient descent with regularization
            gradient = 1/n_data*(X.T@(y-t)) + self.lamb/n_data * self.weight
            self.weight = self.weight - self.lr * gradient

            # cost function with regularization
            loss = 1/-n_data * np.sum(t*np.log(y)) + \
                self.lamb/(2*n_data)*np.sum(self.weight.T@self.weight)

            y_pred = np.argmax(y, axis=1)
            acc_train = self.accuracy_score(y_pred, target)
            print('loss_train: {}, acc_train: {}'.format(
                round(loss, 3), round(acc_train, 3)))
        return self

    def softmax(self, x, weight):
        scores = x@weight
        scores -= np.max(scores, axis=1).reshape((-1, 1))
        return (np.exp(scores)/np.exp(scores).sum(axis=1).reshape((-1, 1)))

    def predict(self, X):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))

        else:
            raise Exception("Input is in Incorrect Format!!")
        return self.softmax(X, self.weight)

    # one hot encoder
    def dummy_gen(self, target):
        n_cases = target.shape[0]
        dummy = csr_matrix(
            (np.ones(n_cases), (np.arange(n_cases), target-target.min())))
        dummy = np.array(dummy.todense())
        return dummy

    # accuracy score
    def accuracy_score(self, y_pred, y_true):
        n_samples = len(y_true)
        sum = 0
        for itr in range(n_samples):
            if y_true[itr] == y_pred[itr]:
                sum += 1
        return (sum/n_samples*100)
