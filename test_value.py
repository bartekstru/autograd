import unittest
import math
from test import Value  # Assuming the Value class is in a file named 'test.py'

class TestValue(unittest.TestCase):

    def test_initialization(self):
        v = Value(5.0)
        self.assertEqual(v.data, 5.0)
        self.assertEqual(v.grad, 0.0)
        self.assertEqual(v._op, '')
        self.assertEqual(v._label, '')

    def test_addition(self):
        a = Value(3.0)
        b = Value(2.0)
        c = a + b
        self.assertEqual(c.data, 5.0)
        c.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)

    def test_multiplication(self):
        a = Value(3.0)
        b = Value(2.0)
        c = a * b
        self.assertEqual(c.data, 6.0)
        c.backward()
        self.assertEqual(a.grad, 2.0)
        self.assertEqual(b.grad, 3.0)

    def test_tanh(self):
        a = Value(0.5)
        b = a.tanh()
        self.assertAlmostEqual(b.data, 0.46211715726000974)
        b.backward()
        self.assertAlmostEqual(a.grad, 0.7864477329659274)

    def test_exp(self):
        a = Value(1.0)
        b = a.exp()
        self.assertAlmostEqual(b.data, math.e)
        b.backward()
        self.assertAlmostEqual(a.grad, math.e)

    def test_power(self):
        a = Value(2.0)
        b = a ** 3
        self.assertEqual(b.data, 8.0)
        b.backward()
        self.assertEqual(a.grad, 12.0)

    def test_division(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        self.assertEqual(c.data, 2.0)
        c.backward()
        self.assertAlmostEqual(a.grad, 1/3)
        self.assertAlmostEqual(b.grad, -2/9)

    def test_mixed_operations(self):
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        d = a * b + c / a - b
        self.assertEqual(d.data, 5.0)
        d.backward()
        self.assertEqual(a.grad, 2.0)
        self.assertEqual(b.grad, 1.0)
        self.assertEqual(c.grad, 0.5)

    def test_backward_propagation(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        d = c + b
        e = d.tanh()
        e.backward()
        self.assertAlmostEqual(a.grad, 0.7064715328911681)
        self.assertAlmostEqual(b.grad, 0.9419620438549241)

    def test_scalar_operations(self):
        a = Value(3.0)
        b = 2 + a
        c = 2 * a
        d = a + 2
        e = a * 2
        self.assertEqual(b.data, 5.0)
        self.assertEqual(c.data, 6.0)
        self.assertEqual(d.data, 5.0)
        self.assertEqual(e.data, 6.0)

    def test_negative_values(self):
        a = Value(-2.0)
        b = Value(3.0)
        c = a + b
        self.assertEqual(c.data, 1.0)
        c.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)

    def test_zero_division(self):
        a = Value(1.0)
        b = Value(0.0)
        with self.assertRaises(ZeroDivisionError):
            c = a / b

    def test_repr(self):
        v = Value(3.14)
        self.assertEqual(repr(v), "Value(data=3.14)")

if __name__ == '__main__':
    unittest.main()