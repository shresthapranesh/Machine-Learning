import numpy as np
from utils import im2col_indices  # CS231N - Stanford
import time


class Conv2D:
    def __init__(self, num_filters, kernel_size, stride=(1, 1), padding=0, lr=0.3):
        self.lr = lr
        self.num_filters = num_filters
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h, self.stride_w = stride
        self.padding = padding
        rng = np.random.default_rng()
        lowerlimit = -np.sqrt(6/(self.kernel_h+self.kernel_w))
        upperlimit = np.sqrt(6/(self.kernel_h+self.kernel_w))
        self.filters = rng.uniform(low=lowerlimit, high=upperlimit,
                                   size=(num_filters, kernel_size[0], kernel_size[1]))
        self.filters = self.filters[:, np.newaxis, :, :]

    def forward_pass(self, input):

        self.last_input = input
        N, D, H, W = input.shape
        h_out = (H + 2 * self.padding -
                 self.kernel_h) // self.stride_h + 1
        w_out = (W + 2 * self.padding - self.kernel_w) // self.stride_w + 1

        self.select_img = im2col_indices(
            input, self.kernel_h, self.kernel_w, self.padding, self.stride_h)
        weights = self.filters.reshape(self.num_filters, -1)
        convolve = weights@self.select_img
        convolve = convolve.reshape(self.num_filters, h_out, w_out, N)
        convolve = convolve.transpose(3, 0, 1, 2)

        return convolve

    def backprop(self, dloss):
        db = np.sum(dloss, axis=(0, 2, 3))
        db = db.reshape(self.num_filters, -1)

        dout_reshaped = dloss.transpose(
            1, 2, 3, 0).reshape(self.num_filters, -1)

        dW = dout_reshaped @ self.select_img.T
        dW = dW.reshape(self.filters.shape)
        self.filters -= self.lr * dW
        return None

    def relu(self, x):
        return np.maximum(0, x)
