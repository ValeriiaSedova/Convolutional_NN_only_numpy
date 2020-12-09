class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.random(self.convolution_shape)
        self.bias = np.random.random()

    def forward(self, input_tensor):
        pass
