import numpy as np

class Constant:
    
    def __init__(self, val=0.1):
        self.val = val
        
    def initialize(self, weight_shape, fan_in, fan_out):
        # print(weight_shape, fan_in.shape, fan_out.shape)
        init_tensor = np.ones(weight_shape) * self.val
        return init_tensor

class UniformRandom:
    
    def initialize(self, weight_shape, fan_in, fan_out):
        init_tensor = np.random.randn(weight_shape)
        return init_tensor

class Xavier:
    def initialize(self, weight_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_out+fan_in))
        init_tensor = np.random.randn(weight_shape)*sigma
        return init_tensor

class He:
    def initialize(self, weight_shape, fan_in, fan_out):
        sigma = np.sqrt(2/fan_in)
        init_tensor = np.random.randn(weight_shape)*sigma
        return init_tensor

