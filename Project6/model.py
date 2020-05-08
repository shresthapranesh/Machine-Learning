import numpy as np
import time
import statistics


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward_pass(output)

        return output

    def backprop(self, label):
        gradient = label
        for i in range(len(self.layers)-1, 0, -1):
            gradient = self.layers[i].backprop(gradient)

        return gradient

    def fit(self, X, label, epochs=10, batch_size=32):
        assert X.shape[0] == label.shape[0]

        no_of_batch = X.shape[0] // batch_size
        X_train = np.array_split(X, no_of_batch)
        y_train = np.array_split(label, no_of_batch)
        batch_loss = []
        for i in range(epochs):
            tic = time.time()
            for X, y in zip(X_train, y_train):
                loss = self.train(X, y.T)
                batch_loss.append(loss)
            avg_loss = statistics.mean(batch_loss)
            toc = time.time()
            print(
                f'epochs: {i+1}/{epochs} Loss: {avg_loss}, time elapsed: {round(toc-tic,4)} sec')

    def train(self, X, label):
        output = self.forward(X)
        self.backprop(label)
        loss = -np.sum(label*np.log(output))*1/output.shape[1]
        return loss

    def predict(self, X):
        return self.forward(X)
