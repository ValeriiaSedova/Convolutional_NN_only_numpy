import numpy as np
from scipy import signal

class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.random([num_kernels, *self.convolution_shape])
        self.bias = np.random.random(num_kernels)
        self._optimizer = None
        self._optimizer_b = None
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
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
        
        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):

        # Stride
        b, d, y, x = error_tensor.shape
        if len(self.convolution_shape) == 1:
            upsampled = np.zeros([b, d, self.input_tensor.shape[2]])
            upsampled[:, :, ::self.stride_shape[0]] = error_tensor
        else:
            upsampled = np.zeros([b, d, self.input_tensor.shape[2], self.input_tensor.shape[3]])
            upsampled[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor

        new_error_tensor = np.zeros(self.input_tensor.shape)
        batch_num = self.input_tensor.shape[0]
        channel_num = self.input_tensor.shape[1]
        gradient_weights = np.zeros(self.weights.shape)
        # print(self.weights.shape)
        # print(error_tensor.shape)

        # Grdaient with respect to the input
        back_weights = np.swapaxes(self.weights, 0, 1)

        for b in range(batch_num):
            for c in range(channel_num):
                for l in range(self.num_kernels): # maybe we need to flip the channels
                    # TODO: arr3[0,:,:] = arr2
                    try:
                        new_error_tensor[b,c] += signal.convolve(upsampled[b,l], back_weights[c,l], mode='same')
                    except:
                        print(upsampled[b,l].shape, back_weights[c,l].shape)
                        # w = np.zeros([0,*self.weights[:].shape])
                        # w[0,:,:] = self.weights[:,c]
                        # new_error_tensor[b,c] += signal.convolve(error_tensor[b,l], w, mode='same')

        # Gradient with respect to the weights
        # S = self.input_tensor.shape[1]
        # for b in range(batch_num):
        #     for c in range(channel_num):  # channel = kernel
        #         for s in range(S):        
        #             padded_channel = np.pad(self.input_tensor[b,s], self.weights.shape[2]//2)
        #             gradient_weights[c, s] = signal.correlate(padded_channel, error_tensor[b, c], mode='valid')
        # gradient_bias = error_tensor.sum(axis=0,keepdims=True)
        # # Update weights and bias
        # if self._optimizer != None:
        #     self.weights = self._optimizer.calculate_update(self.weights, gradient_weights)

        # if self._optimizer_b != None:
        #     self.bias = self._optimizer_b.calculate_update(self.bias, gradient_bias)

        return new_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in    = np.prod(self.convolution_shape)
        fan_out   = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def optimizer_b(self):
        return self._optimizer_b

    @optimizer_b.setter
    def optimizer_b(self, optimizer_b):
        self._optimizer_b = optimizer_b

