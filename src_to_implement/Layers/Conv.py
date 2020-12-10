import numpy as np

class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.random(num_kernels, self.convolution_shape)
        self.bias = np.random.random(num_kernels)

    def forward(self, input_tensor):
        for element in range(input_tensor.shape[0]):
            for kernel in range(self.num_kernels):
                pass


        
