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
        batch_num = input_tensor.shape[0]
        channel_num = input_tensor.shape[1]
        ONEDCONV = len(input_tensor.shape) == 3
        output_tensor = np.zeros([batch_num, self.num_kernels, *input_tensor.shape[2:]])
        
        for b in range(batch_num):            # iterate over each tensor in the batch
            for k in range(self.num_kernels): # iterate over each kernel
                for c in range(channel_num):  # iterate over each channel to sum them up in the end to get 3D convolution (feature map)
                    output_tensor[b, k] += signal.correlate(input_tensor[b, c], self.weights[k, c], mode = 'same')
        
                output_tensor[b,k] += self.bias[k] # add bias to each feature map

        # stride 
        if ONEDCONV:
            output_tensor = output_tensor[:, :, ::self.stride_shape[0]]
        else:
            output_tensor = output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

        return output_tensor
