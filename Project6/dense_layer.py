import numpy as np
import time
# Part A


class FullyConnected:
    '''
    Your regular dense layer
    '''

    def __init__(self, layers, reg=1e-3, lr=0.3):
        self.layers = layers
        self.W = {}
        self.B = {}
        self.A = {}
        self.H = {}
        self.dW = {}
        self.dB = {}
        self.dH = {}
        self.dA = {}
        self.lr = lr
        self.reg = reg
        self.build_check = False

    def build(self, x):
        self.layers.insert(0, x.shape[0])
        self.input_size = x.shape[1]
        self.n_layers = len(self.layers)
        rng = np.random.default_rng()
        for i in range(len(self.layers)-1):
            lowerlimit = -np.sqrt(6/(self.layers[i+1]+self.layers[i]))
            upperlimit = np.sqrt(6/(self.layers[i+1]+self.layers[i]))
            self.W[i +
                   1] = rng.uniform(low=lowerlimit, high=upperlimit, size=(self.layers[i+1], self.layers[i]))
            self.B[i+1] = np.zeros((self.layers[i+1], 1))
        self.build_check = True

    def forward_pass(self, x):

        self.d_input_shape = x.shape
        x = x.reshape(x.shape[0], -1).T
        self.H[0] = x
        if self.build_check == False:
            self.build(x)
        for i in range(self.n_layers-2):
            self.A[i+1] = self.W[i+1]@self.H[i]+self.B[i+1]
            self.H[i+1] = self.hidden_Activation(self.A[i+1])

        self.A[self.n_layers-1] = self.W[self.n_layers-1]@self.H[self.n_layers-2] + \
            self.B[self.n_layers-1]
        self.H[self.n_layers -
               1] = self.output_Activation(self.A[self.n_layers-1])

        return self.H[self.n_layers-1]

    def backprop(self, y):
        for k in range(self.n_layers-1, 0, -1):
            if k == self.n_layers-1:
                self.dA[k] = self.H[k]-y
            else:
                self.dA[k] = self.dH[k] * self.hidden_dActivation(self.H[k])
            self.dH[k-1] = self.W[k].T@self.dA[k]
            self.dW[k] = self.dA[k]@self.H[k-1].T * 1/self.input_size
            self.dW[k] += self.reg * self.W[k]
            self.dB[k] = np.sum(
                self.dA[k], axis=1, keepdims=True) * 1/self.input_size
            self.dB[k] += self.reg * self.B[k]

        for i in range(self.n_layers-1):
            self.W[i+1] -= self.lr * self.dW[i+1]
            self.B[i+1] -= self.lr * self.dB[i+1]

        return self.dH[0].reshape(self.d_input_shape)

    def hidden_Activation(self, scores):
        return np.maximum(0, scores)

    def hidden_dActivation(self, scores):
        x = np.zeros_like(scores)
        x[scores > 0] = 1
        return x

    def output_Activation(self, scores):
        scores -= np.max(scores, axis=0)
        exp = np.exp(scores)
        return exp / np.sum(exp, axis=0)

    def output_dActivation(self, scores):
        return -np.outer(scores, scores)+np.diag(scores.flatten())
