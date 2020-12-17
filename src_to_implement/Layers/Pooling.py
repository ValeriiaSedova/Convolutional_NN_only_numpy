import numpy as np

class Pooling:

    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_num = input_tensor.shape[0]
        channel_num = input_tensor.shape[1]
        hi, wi = input_tensor.shape[2:]
        h, w = self.pooling_shape
        output_tensor = np.zeros([])

        for b in 
