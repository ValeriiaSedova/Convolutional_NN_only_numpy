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
        # print(input_tensor.shape)
        ONEDIM = len(input_tensor.shape) == 3
        if ONEDIM:
            batch, channels, y = input_tensor.shape
            output_tensor = np.zeros([batch, self.num_kernels, y])            
        else:
            batch, channels, y, x = input_tensor.shape
            output_tensor = np.zeros([batch, self.num_kernels, y, x])
        if type(self.stride_shape) == tuple:
            cc, cy, cx = self.convolution_shape
        else:
            cc = self.convolution_shape
            cx = cc; cy = cc;

        for element in range(batch):
            for k in range(self.num_kernels):
                kernel = self.weights[k]
                if ONEDIM: loc_maps = np.zeros([channels, y])
                else: loc_maps = np.zeros([channels, y, x])

                for channel in range(channels):
                    local = input_tensor[element, channel]
                    if ONEDIM: loc_maps[channel] = local * kernel[channel]
                    else: loc_maps[channel] = signal.convolve2d(kernel[channel], local, mode = 'same')
                
                if ONEDIM: feature_map = loc_maps[::cc, ::cy, ::cx].sum(axis = 0) + self.bias[channel]
                else: feature_map = loc_maps[::cc, ::cy].sum(axis = 0) + self.bias[channel]
                output_tensor[element, k] = feature_map
        
        return output_tensor

                    




        
