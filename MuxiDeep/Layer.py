if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from MuxiDeep import Layer
from MuxiDeep import Parameter
import functions as F
import numpy as np

class Linear(Layer):
    def __init__(self, out_size, nobias = False, dtype = np.float32, in_size = None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self.__init__W(in_size)

        self.dtype = dtype
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size,dtype=dtype), name='b')

    def __init__W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, inputs):
        if self.W.data is None:
            self.in_size = inputs.shape[1]
            self.__init__W()
        
        y = F.linear(inputs, self.W, self.b)
        return y