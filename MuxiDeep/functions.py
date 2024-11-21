if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from MuxiDeep import Function
from MuxiDeep import broadcast_to
import numpy as np

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0]
        gx = exp(x) * gy
        return gx
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = cos(x) * gy
        return gx
    
class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = x * sin(x)
        return gx
    

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gw = matmul(x.T, gy)
        return gx, gw
    
    
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = - gx0
        return gx0, gx1
    
class Linear(Function):
    def forward(self, x, W, b=None):
        t = matmul(x, W)
        if b is None:
            return t
        
        y = t + b
        t = None #引用计数减为0，删除t中的数据
        return y

    def backward(self, gy):
        x, W= self.inputs[0], self.inputs[1]
        gx = matmul(gy, W.T)
        gw = matmul(x.T, gy)
        gb = gy
        return gx, gw, gb
    
class Sigmoid(Function):
    def forward(self, x):
        x = self.inputs
        y = 1 / (1 + exp(-x))
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx
    
def sigmoid(x):
    return Sigmoid()(x)

def linear(x, W, b):
    return Linear()(x, W, b)

def mean_square_error(x0, x1):
    return MeanSquaredError()(x0, x1)
    

def matmul(x, W):
    return MatMul()(x, W)

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)

def exp(x):
    return Exp()(x)