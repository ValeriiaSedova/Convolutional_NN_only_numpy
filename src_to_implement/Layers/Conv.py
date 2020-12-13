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
        print(input_tensor[0].shape, self.convolution_shape)
        if len(self.convolution_shape) == 2:
            ONEDIM = self.convolution_shape[1] == 1
        else:
            ONEDIM = self.convolution_shape[1] == 1 and self.convolution_shape[2] == 1
            
        if len(self.stride_shape) == 2:
            sy, sx = self.stride_shape
        else:
            sy = self.stride_shape[0]
            sx = sy
        
        y, x = input_tensor.shape[-2], input_tensor.shape[-1]
        if sy%2 == 0:
            out_y = y//sy
        else:
            out_y = y//sy + 1
        
        if sx%2 == 0:
            out_x = x//sx
        else:
            out_x = x//sx + 1

        if len(input_tensor.shape) == 4:
            b, c, _, _ = input_tensor.shape
            output_tensor = np.zeros([b, self.num_kernels, out_y, out_x])
        else:
            b, c, _ = input_tensor.shape
            output_tensor = np.zeros([b, self.num_kernels, out_y])

        for element in range(b):
            for ki in range(self.num_kernels):
                feature_map = np.zeros(input_tensor[element, 0].shape)
                KONEDIM = len(self.weights[ki, 0].shape) == 1
                for channel in range(c):
                    if ONEDIM:
                        feature_map += input_tensor[element, channel]*self.weights[ki,channel]
                    elif KONEDIM:
                        feature_map += signal.correlate(input_tensor[element, channel],
                                                        self.weights[ki, channel], mode='same')
                    else:
                        feature_map += signal.correlate2d(input_tensor[element, channel],
                                                    self.weights[ki, channel], mode='same') 

                if ONEDIM:
                    output_tensor[element, ki] = feature_map[::sy, ::sx] + self.bias[ki]
                elif KONEDIM:
                    output_tensor[element, ki] = feature_map[::sy] + self.bias[ki]
                else:
                    output_tensor[element, ki] = feature_map[::sy, ::sx] + self.bias[ki]
        return output_tensor










        # if len(input_tensor.shape) == 4:
        #     batch, channels, y, x = input_tensor.shape
        #     output_tensor = np.zeros([batch, self.num_kernels, y, x])
        # else:
        #     batch, channels, y = input_tensor.shape
        #     output_tensor = np.zeros([batch, self.num_kernels, y])
        # if type(self.stride_shape) == tuple:
        #     cc, cy, cx = self.convolution_shape
        # else:
        #     cc = self.convolution_shape
        #     cx = cc; cy = cc;

        # for element in range(batch):
        #     for k in range(self.num_kernels):
        #         kernel = self.weights[k]
        #         if ONEDIM: loc_maps = np.zeros([channels, y])
        #         else: loc_maps = np.zeros([channels, y, x])

        #         for channel in range(channels):
        #             local = input_tensor[element, channel]
        #             if ONEDIM: loc_maps[channel] = local * kernel[channel]
        #             else: loc_maps[channel] = signal.convolve2d(kernel[channel], local, mode = 'same')
                
        #         if ONEDIM: feature_map = loc_maps[::cc, ::cy, ::cx].sum(axis = 0) + self.bias[channel]
        #         else: feature_map = loc_maps[::cc, ::cy].sum(axis = 0) + self.bias[channel]
        #         output_tensor[element, k] = feature_map
        
        # return output_tensor

                    




        
