if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import MuxiDeep as md
import numpy as np
import MuxiDeep.functions as F


def numerical_diff(f,x,eps=1e-4):
    x0 = md.Variable(md.as_array(x.data - eps))
    x1 = md.Variable(md.as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = md.Variable(np.array(2.0))
        y = md.square(x)
        excepted = np.array(4.0)
        self.assertEqual(y.data, excepted)

    def test_complex_double(self):
        x = md.Variable(np.array(2.0))
        a = md.square(x)
        y = md.square(a) + md.square(a) + 9 * x
        y.backward()
        excepted = 73.0
        self.assertEqual(x.grad.data, excepted)


class diffTest(unittest.TestCase):
    def test_Exp(self):
        x = md.Variable(np.array(4))
        y = md.exp(x)
        y.backward()
        num_grad = numerical_diff(md.exp, x)
        flg = np.allclose(x.grad.data, num_grad)
        self.assertTrue(flg)

    def test_Squre(self):
        x = md.Variable(np.array(5))
        y = md.square(x)
        y.backward()
        num_grad = numerical_diff(md.square, x)
        flg = np.allclose(x.grad.data, num_grad)
        self.assertTrue(flg)

    def test_Sub(self):
        x = md.Variable(np.array(5))
        y = 2.0 - x
        excepted = np.array(-3.0)
        y1 = x - 2.0
        except2 = np.array(3.0)
        self.assertEqual(y.data, excepted)
        self.assertEqual(y1.data, except2)

    def test_goldstein(self):
        x = md.Variable(np.array(1.0))
        y = md.Variable(np.array(1.0))
        z = goldstein(x, y)
        z.backward()
        except1 = -5376.0
        except2 = 8064.0
        self.assertEqual(x.grad.data, except1)
        self.assertEqual(y.grad.data, except2)

class broadcastTest(unittest.TestCase):
    def test_bs_add(self):
        x0 = md.Variable(np.array([1,2,3]))
        x1 = md.Variable(np.array([10]))
        y = x0 + x1
        y.backward()
        except1 = np.array([11,12,13])
        except2 = np.array([3])
        self.assertTrue(np.array_equal(y.data, except1))
        self.assertTrue(np.array_equal(x1.grad.data, except2))

    def test_linear(self):
        np.random.seed(0)
        x = np.random.rand(100,1)
        y = 5 + 2 * x + np.random.rand(100, 1)

        W = md.Variable(np.zeros((1,1)))
        b = md.Variable(np.zeros(1))

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
        except1 = np.array([[2.11807369]])
        except2 = np.array([[5.46608905]])
        self.assertTrue(np.allclose(W.data, except1))
        self.assertTrue(np.allclose(b.data, except2))
unittest.main()
