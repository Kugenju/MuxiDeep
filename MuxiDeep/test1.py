import unittest
import Core as md
import numpy as np

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = md.Variable(np.array(2.0))
        y = md.square(x)
        excepted = np.array(4.0)
        self.assertEqual(y.data, excepted)

    def test_complex_double(self):
        x = md.Variable(np.array(2.0))
        a = md.square(x)
        y = md.add(md.square(a),md.square(a))
        y.backward()
        excepted = 64.0
        self.assertEqual(x.grad, excepted)


class diffTest(unittest.TestCase):
    def test_Exp(self):
        x = md.Variable(np.array(4))
        y = md.exp(x)
        y.backward()
        num_grad = md.numerical_diff(md.exp, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

    def test_Squre(self):
        x = md.Variable(np.array(5))
        y = md.square(x)
        y.backward()
        num_grad = md.numerical_diff(md.square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


unittest.main()
