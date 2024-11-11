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

# endregion

class Config:
    enable_backprop = True #是否开启反向传播

class Variable:

    __array_priority__ = 200 # 提高Variable类中运算符方法在实际运算时的优先级，以实现和ndarray同时进行计算

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

    def backward(self, retain_grad = False):
        funcs = [ ]
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                heapq.heappush(funcs,(-f.generation, f))#使用优先队列管理函数列表
                seen_set.add(f)

        add_func(self.creator)

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        
        while funcs:
            f = heapq.heappop(funcs)[1]
            gys = [output().grad for output in f.outputs]
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
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        #实现print(Variable)的效果，同时要支持数据为None和分行输出
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

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


# region define function class
class Add(Function):
    
    def forward(self,x0,x1):
        y = x0 + x1
        return y
    
    def backward(self,gy):
        return gy, gy

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx
    
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1 * gy, x0 * gy
    
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy/x0
        gx1 = - gy * (x0 / x1 ** 2)
        return gx0, gx1
    
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx
# endregion    

# region define function
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
    b = square(a) + square(a)

    c = a * b + x
    c.backward()
    print(c.data,b.data,a.data)
    print(c.grad,b.grad, a.grad, x.grad)
    print(-x)
    print(x ** 3)