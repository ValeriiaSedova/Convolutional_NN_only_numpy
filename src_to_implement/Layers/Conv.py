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
        #backward pass the error
        output = np.zeros(self.input_tensor.shape)
        gradtensor = np.zeros(self.weights.shape)
        self.gradient_tensor = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        new_error_tensor = np.zeros((error_tensor.shape[0], self.num_kernels, *self.input_tensor.shape[2:] ))



        #check the change of weight tensor before you convolve
        for batch in range(error_tensor.shape[0]):
            #strides
            error_tensor_strided = np.zeros((self.num_kernels, *self.input_tensor.shape[2:] ))
            for kernels in range(error_tensor.shape[1]):
                errorimage = error_tensor[batch, kernels, :]
                if len(error_tensor.shape)==4:
                    error_tensor_strided[kernels,:][np.s_[::self.stride_shape[0]], :][:, np.s_[::self.stride_shape[1]]] = errorimage
                else:
                    error_tensor_strided[kernels,:][np.s_[::self.stride_shape[0]]] = errorimage

            #output
            for channels in range(self.weights.shape[1]):
                err = signal.convolve(error_tensor_strided, np.flip(self.weights, 0)[:,channels,:], mode='same')
                midchannel = int(err.shape[0] / 2)
                op = err[midchannel,:]
                output[batch,channels,:] = op    #numkernel and numchannel inetrchange here numkernel is one dim depth

            for kernels in range(error_tensor_strided.shape[0]):
                self.grad_bias[kernels] += np.sum(error_tensor[batch, kernels, :]) #bias is sum of error tensors

                for channels in range(self.input_tensor.shape[1]):
                    inputimg = self.input_tensor[batch,channels,:]
                    if len(error_tensor.shape)==4:
                        padx = self.convolution_shape[1]/2
                        pady = self.convolution_shape[2]/2
                        padimg = np.pad(inputimg, ((int(np.floor(padx)),int(np.floor(padx-0.5))),(int(np.floor(pady)),int(np.floor(pady-0.5)))), mode="constant")
                    else:
                        padx = self.convolution_shape[1]/2
                        padimg = np.pad(inputimg, ((int(np.floor(padx)),int(np.floor(padx-0.5)))), mode="constant")
                    gradtensor[kernels,channels,:] = signal.correlate(padimg, error_tensor_strided[kernels,:], mode="valid")

            #print(error_tensor_strided)
            self.gradient_tensor += gradtensor


        #update weights
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_tensor)
        if self._optimizer_b is not None:
            self.bias = self._optimizer_b.calculate_update(self.bias, self.grad_bias)

        return output

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

    @property
    def gradient_weights(self):
        return self.gradient_tensor

    @property
    def gradient_bias(self):
        return self.grad_bias
