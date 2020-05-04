import numpy as np
from utils import im2col_indices, col2im_indices  # CS231N - Stanford
import time


class Maxpool2D:
    def __init__(self, pool=(2, 2), stride=(2, 2), padding=None):
        self.stride_h, self.stride_w = stride
        self.pool_h, self.pool_w = pool
        if padding is not None:
            self.pad_h, self.pad_w = padding
        else:
            self.pad_h, self.pad_w = (0, 0)

    def forward_pass(self, input):
        self.last_input_shape = input.shape
        N, D, H, W = input.shape
        h_out = 1 + (H+2*self.pad_h-self.pool_h)//self.stride_h
        w_out = 1 + (W+2*self.pad_w-self.pool_w)//self.stride_w

        assert self.pool_h == self.pool_w == self.stride_w, 'This method requires pooling with equal size'
        assert H % self.pool_h == 0, 'check params'
        assert W % self.pool_w == 0, 'check params'
        self.x_reshaped = input.reshape(
            N, D, H//self.pool_h, self.pool_h, W//self.pool_w, self.pool_w)
        self.out = self.x_reshaped.max(axis=3).max(axis=4)
        return self.relu(self.out)

    def backprop(self, dout):

        dx_reshaped = np.zeros_like(self.x_reshaped)
        out_newaxis = self.out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (self.x_reshaped == out_newaxis)
        dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_reshaped.reshape(self.last_input_shape)

        return dx

    def relu(self, x):
        return np.maximum(0, x)
