import numpy as np


class Conv2D:
    def __init__(self, num_filters, kernel_size, stride=1, padding=0, lr=0.3):
        self.lr = lr
        self.num_filters = num_filters
        self.kernel_h, self.kernel_w = kernel_size
        self.stride = stride
        self.padding = padding
        rng = np.random.default_rng()
        lowerlimit = -np.sqrt(6/(self.kernel_h+self.kernel_w))
        upperlimit = np.sqrt(6/(self.kernel_h+self.kernel_w))
        self.filters = rng.uniform(low=lowerlimit, high=upperlimit,
                                   size=(num_filters, kernel_size[0], kernel_size[1]))
        self.filters = self.filters[:, np.newaxis, :, :]
        self.bias = np.zeros(self.num_filters)
        self.m = 0
        self.v = 0
        self.beta1, self.beta2 = (0.9, 0.999)
        self.itr = 0

    def forward_pass(self, input):

        self.last_input = input
        N, D, H, W = input.shape
        h_out = (H + 2 * self.padding -
                 self.kernel_h) // self.stride + 1
        w_out = (W + 2 * self.padding - self.kernel_w) // self.stride + 1
        shape = (D, self.kernel_h, self.kernel_w, N, h_out, w_out)
        strides = (H*W, W, 1, D*H*W, self.stride*W, self.stride)
        strides = input.itemsize * np.array(strides)
        input_stride = np.lib.stride_tricks.as_strided(
            input, shape=shape, strides=strides)
        self.x_cols = np.ascontiguousarray(input_stride)
        self.x_cols.shape = (D*self.kernel_h*self.kernel_w, N*h_out*w_out)
        res = self.filters.reshape(
            self.num_filters, -1)@self.x_cols + self.bias.reshape(-1, 1)
        res.shape = (self.num_filters, N, h_out, w_out)
        self.out = res.transpose(1, 0, 2, 3)
        self.out = np.ascontiguousarray(self.out)

        return self.relu(self.out)

    def backprop(self, dloss):
        self.itr += 1
        dloss = dloss * self.d_relu(self.out)
        db = np.sum(dloss, axis=(0, 2, 3))
        db = db.reshape(self.num_filters, -1)

        dout_reshaped = dloss.transpose(
            1, 2, 3, 0).reshape(self.num_filters, -1)

        dW = dout_reshaped @ self.x_cols.T
        dW = dW.reshape(self.filters.shape)
        self.m = self.beta1 * self.m + \
            (1-self.beta1) * self.filters
        self.v = self.beta2 * self.v + \
            (1-self.beta2) * np.square(self.filters)
        mt = self.m/(1-self.beta1**self.itr)
        vt = self.v/(1-self.beta2**self.itr)
        self.filters -= self.lr * mt / (np.sqrt(vt)+1e-7)
        self.bias -= self.lr * db

        return None

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        x = np.zeros_like(x)
        x[x > 0] = 1
        return x
