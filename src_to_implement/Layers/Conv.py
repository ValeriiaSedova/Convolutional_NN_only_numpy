import numpy as np
from scipy import signal

class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.random([num_kernels, *self.convolution_shape])
        self.bias = np.random.random(num_kernels)
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_shape = input_tensor.shape

        self.batch_size = input_tensor.shape[0]
        num_channels = input_tensor.shape[1]
        summation = np.zeros((self.batch_size, self.num_kernels, *self.input_shape[2:]))

        if len(self.input_tensor.shape) == 3:
            self.width = input_tensor.shape[2]

        else:

            self.width = input_tensor.shape[2]
            self.height = input_tensor.shape[3]

        for i in range(self.batch_size):
            for j in range(self.num_kernels):
                for k in range(num_channels):
                    summation = summation + signal.correlate(input_tensor[i, k, :], self.weights[j, k, :], mode='same')

                summation = summation + self.bias[j]

        if len(self.input_tensor.shape) == 3:

            summation = summation[:, :, 0: self.width: self.stride_shape[0]]

        else:
            summation = summation[:, :, 0: self.width: self.stride_shape[0], 0: self.height: self.stride_shape[1]]

        return summation