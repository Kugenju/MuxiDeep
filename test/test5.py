if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from MuxiDeep import Variable
import numpy as np
import MuxiDeep.functions as F

np.random.seed(0)
x = np.random.rand(100,1)
y = 5 + 2 * x + np.random.rand(100, 1)

W = Variable(np.zeros((1,1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_square_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)