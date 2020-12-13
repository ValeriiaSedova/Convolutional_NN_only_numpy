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
        batch, channels, y, x = input_tensor.shape
        output_tensor = np.zeros([batch, self.num_kernels, y, x])
        # if type(self.stride_shape) == tuple:
        #     cc, cy, cx = self.convolution_shape
        for element in range(batch):
            for k in range(self.num_kernels):
                kernel = self.weights[k]
                loc_maps = np.zeros(channels, y, x)

                for channel in range(channels):
                    local = input_tensor[element, channel]
                    loc_maps[channel] = signal.convolve2d(kernel[channel], local, mode = 'same')
                
                feature_map = loc_maps.sum(axis = 0) + self.bias[channel]
                output_tensor[element, k] = feature_map
        
        return output_tensor

                    




        
