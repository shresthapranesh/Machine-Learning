import numpy as np


# Part A
class FullyConnected:
    def __init__(self, layers, size, lr=0.3):
        self.layers = layers
        self.input_size = size
        self.W = {}
        self.B = {}
        self.A = {}
        self.H = {}
        self.dW = {}
        self.dB = {}
        self.dH = {}
        self.dA = {}
        self.n_layers = len(layers)
        self.lr = lr
        self.loss_list = []

    def forward_pass(self, x):

        self.H[0] = x
        for i in range(self.n_layers-2):
            self.A[i+1] = self.W[i+1]@self.H[i]+self.B[i+1]
            self.H[i+1] = self.sigmoid(self.A[i+1])

        self.A[self.n_layers-1] = self.W[self.n_layers-1]@self.H[self.n_layers-2] + \
            self.B[self.n_layers-1]
        self.H[self.n_layers-1] = self.sigmoid(self.A[self.n_layers-1])

        return self.H[self.n_layers-1]

    def backprop(self, X, y):

        for k in range(self.n_layers-1, 0, -1):
            if k == self.n_layers-1:
                self.dA[k] = (self.H[k]-y) * self.d_sigmoid(self.H[k])
            else:
                self.dA[k] = self.dH[k] * self.d_sigmoid(self.H[k])
            self.dH[k-1] = self.W[k].T@self.dA[k]
            self.dW[k] = self.dA[k]@self.H[k-1].T * 1./self.H[k-1].shape[1]
            self.dB[k] = self.dA[k]@np.ones(
                [self.dA[k].shape[1], 1]) * 1./self.H[k-1].shape[1]

        for i in range(self.n_layers-1):
            self.W[i+1] -= self.lr * self.dW[i+1]
            self.B[i+1] -= self.lr * self.dB[i+1]

    def fit(self, X, y, epochs=1000):

        rng = np.random.default_rng()

        for i in range(len(self.layers)-1):
            self.W[i +
                   1] = rng.standard_normal(size=(self.layers[i+1], self.layers[i]))
            self.B[i+1] = np.zeros((self.layers[i+1], 1))
        for i in range(epochs):
            self.forward_pass(X)
            self.backprop(X, y)
            a = self.log_loss(self.H[self.n_layers-1], y)
            print('Epoch: {}/{}, Training Loss: {}'.format(i+1, epochs, a))
            self.loss_list.append(a)

    def sigmoid(self, scores):
        return 1/(1+np.exp(-scores))

    def d_sigmoid(self, scores):
        return scores * (1-scores)

    def log_loss(self, y, t):
        loss = 1/-self.input_size * (np.sum(t*np.log(y)))
        return loss

    def predict(self, x):
        return self.forward_pass(x)


# for Part B


class FullyConnectedRegressor:
    def __init__(self, layers, size, lr=0.3):
        self.layers = layers
        self.input_size = size
        self.W = {}
        self.B = {}
        self.A = {}
        self.H = {}
        self.dW = {}
        self.dB = {}
        self.dH = {}
        self.dA = {}
        self.n_layers = len(layers)
        self.lr = lr
        self.loss_list = []

    def forward_pass(self, x):

        self.H[0] = x
        for i in range(self.n_layers-2):
            self.A[i+1] = self.W[i+1]@self.H[i]+self.B[i+1]
            self.H[i+1] = self.tanh(self.A[i+1])

        self.A[self.n_layers-1] = self.W[self.n_layers-1]@self.H[self.n_layers-2] + \
            self.B[self.n_layers-1]
        self.H[self.n_layers-1] = self.tanh(self.A[self.n_layers-1])

        return self.H[self.n_layers-1]

    def backprop(self, X, y):

        for k in range(self.n_layers-1, 0, -1):
            if k == self.n_layers-1:
                self.dA[k] = 2*(self.H[k]-y) * self.d_tanh(self.H[k])
            else:
                self.dA[k] = self.dH[k] * self.d_tanh(self.H[k])
            self.dH[k-1] = self.W[k].T@self.dA[k]
            self.dW[k] = self.dA[k]@self.H[k-1].T * 1./self.H[k-1].shape[1]
            self.dB[k] = self.dA[k]@np.ones(
                [self.dA[k].shape[1], 1]) * 1./self.H[k-1].shape[1]

        for i in range(self.n_layers-1):
            self.W[i+1] -= self.lr * self.dW[i+1]
            self.B[i+1] -= self.lr * self.dB[i+1]

    def fit(self, X, y, epochs=1000):

        rng = np.random.default_rng()

        for i in range(len(self.layers)-1):
            self.W[i +
                   1] = rng.standard_normal(size=(self.layers[i+1], self.layers[i]))
            self.B[i+1] = np.zeros((self.layers[i+1], 1))
        for i in range(epochs):
            self.forward_pass(X)
            self.backprop(X, y)
            a = self.r_loss(self.H[self.n_layers-1], y)
            print('Epoch: {}/{}, Training Loss: {}'.format(i+1, epochs, a))
            self.loss_list.append(a)

    def r_loss(self, y, t):
        loss = 1/self.input_size * (np.sum(np.sqrt(np.square(y-t))))
        return loss

    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        return 1-np.square((x))

    def predict(self, x):
        return self.forward_pass(x)
