import numpy as np
import heapq
import weakref
import contextlib

# region else func
def as_array(input):
    if isinstance(input, Variable):
        return input
    if np.isscalar(input):
        return np.array(input)
    return(input)

def as_variable(input):
    if isinstance(input, Variable):
        return input
    else:
        return Variable(input)

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
        #yield前是预处理部分，yield后是后处理的代码
        #在使用with语句调用该函数的情况下：
        #当处理进入with块的作用域时执行预处理的代码，当处理离开with块的作用域时执行后处理的代码
    finally:
        setattr(Config, name, old_value)

def no_grad():
    #设置禁用反向传播， 通过with no_grad():使用
    return using_config('enable_backprop', False)

def Sum_to(x, shape):
    current_shape = x.shape

    if current_shape == shape:
        return x
    
    axes = tuple(i for i, (dim_x, dim_shape) in enumerate(zip(current_shape, shape)) if dim_x != dim_shape)

    result = np.sum(x, axis=axes, keepdims=True)
    return result.reshape(shape)
# endregion

class Config:
    enable_backprop = True #是否开启反向传播

class Variable:

    __array_priority__ = 200 
    # 提高Variable类中运算符方法在实际运算时的优先级，以实现和ndarray同时进行计算

    def __init__(self, data,name = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.name = name
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad = False, create_graph = False):
        funcs = [ ]
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                heapq.heappush(funcs,(-f.generation, f))#使用优先队列管理函数列表
                seen_set.add(f)

        add_func(self.creator)

        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        
        while funcs:
            f = heapq.heappop(funcs)[1]
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)

                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)
        
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None
    
    def cleargrad(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def T(self):
        return transpose(self)
    
    def transpose(self):
        return transpose(self)

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        #实现print(Variable)的效果，同时要支持数据为None和分行输出
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)
    
    def sum(self):
        return sum(self)


class Function:
    def __call__(self, *inputs, name = None):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)#使用星号解包

        if not isinstance(ys, tuple):
            ys = (ys,) #处理只有一个返回值的情况

        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
        self.name = name
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self,x):
        raise NotImplementedError()
    
    def backward(self,gy):
        raise NotImplementedError()
    
    def __lt__(self, other):
        return self.generation < other.generation

class Parameter(Variable):
    pass

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name: str, value):
        # 只有Parameter类的实例会被储存为参数
        if isinstance(value, Parameter):
            self._params.add(name)
        super.__setattr__(name, value)
        #layer.p1 = Parameter(np.array(1.0))

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)#处理返回值为一个的情况
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
        

    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            yield self.__dict__[name]
            #yield类似于return，区别是yield会暂停处理并返回值，在此使用yield语句会恢复执行处理

    def cleangrads(self):
        for param in self.params():
            param.cleangrad()

# region define function class
class Add(Function):
    
    def forward(self,x0,x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    
    def backward(self,gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gy, self.x0_shape)
            gx1 = sum_to(gy, self.x1_shape)
        return gx0, gx1

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0]
        gx = exp(x) * gy
        return gx
    
class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0:2]
        gx0, gx1= x1 * gy, x0 * gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1
    
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0:2]
        gx0 = gy/x0
        gx1 = - gy * (x0 / x1 ** 2)
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1
    
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx
    
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    
    def backward(self, gy):
        gx = np.transpose(gy)
        return gx
    
class BroadCast_To(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, shape=self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

class Sum_To(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = Sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
class Sum(Function):
    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum()
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
# endregion    

# region define function
def sum(x):
    return Sum()(x)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

def pow(x, c):
    return Pow(c)(x)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

def transpose(x):
    return Transpose()(x)

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadCast_To(shape)(x)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Sum_To(shape)(x)
# endregion
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__div__ = div
    Variable.__rdiv__ = rdiv
    Variable.__pow__ = pow
#主函数
if __name__ == '__main__':
    setup_variable()
    x = Variable(np.array(2.0))
    a = square(x)
    x = Variable(np.random.randn(2,2,3))
    #print(x)
    y = x.reshape((4,3))
    #print(y)
    y.backward(retain_grad=True)
    #print(z)
    print(x.grad)