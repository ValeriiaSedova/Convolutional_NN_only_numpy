import numpy as np

class Pooling:

    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def pooling(self, ps, stride, a):
        if ps[0]%2 == 0: w = int(a.shape[0]-np.ceil((ps[0]-1)/2))
        if ps[1]%2 == 0: h = int(a.shape[1]-np.ceil((ps[1]-1)/2))
        if ps[0]%2 != 0: w = int(a.shape[0]-ps[0]//2)
        if ps[1]%2 != 0: h = int(a.shape[1]-ps[1]//2)

        o = np.zeros([w,h])

        for (y,x), _ in np.ndenumerate(a):
            if y<=a.shape[0]-ps[0] and x<=a.shape[1]-ps[1]:
                mx = a[y:y+ps[0], x:x+ps[1]].max()
                o[y,x] = mx
        
        so = np.zeros([o.shape[0]//stride[0], o.shape[1]//stride[1]])
        so = o[::stride[0],::stride[1]]
        return so

    def upsampling(self, ps, stride, e, a):
        o = np.zeros_like(a)

        for (y, x), v in np.ndenumerate(e):
            ay = y*stride[0]
            ax = x*stride[1]
            
            loc = a[ay:ay+ps[0],ax:ax+ps[1]]
            i,j = np.where(loc == v)
            o[ay+i, ax+j] += v

        return o

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_num = input_tensor.shape[0]
        channel_num = input_tensor.shape[1]
        output_tensor = []
        for b in range(batch_num):
            loc = []
            for c in range(channel_num):
                loc.append(self.pooling(self.pooling_shape,
                                        self.stride_shape,
                                        input_tensor[b,c]))
            output_tensor.append(loc)
        
        output_tensor = np.array(output_tensor)
        return output_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        batch_num = error_tensor.shape[0]
        channel_num = error_tensor.shape[1]
        output_tensor = []
        for b in range(batch_num):
            loc = []
            for c in range(channel_num):
                loc.append(self.upsampling( self.pooling_shape,
                                            self.stride_shape,
                                            error_tensor[b,c],
                                            self.input_tensor[b,c]))
            output_tensor.append(loc)
        
        output_tensor = np.array(output_tensor)
        return output_tensor

        
