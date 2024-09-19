import unittest
import torch
from micrograd.engine import Value

class TestEngine(unittest.TestCase):

    def test_sanity_check(self):
        # MicroGrad
        x = Value(-4.0)
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xmg, ymg = x, y

        # PyTorch
        x = torch.Tensor([-4.0]).double()
        x.requires_grad = True
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xpt, ypt = x, y

        # Assertions
        self.assertEqual(ymg.data, ypt.data.item())
        self.assertEqual(xmg.grad, xpt.grad.item())

    def test_more_ops(self):
        # MicroGrad
        a = Value(-4.0)
        b = Value(2.0)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f
        g.backward()
        amg, bmg, gmg = a, b, g

        # PyTorch
        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([2.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = a + b
        d = a * b + b**3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g = g + 10.0 / f
        g.backward()
        apt, bpt, gpt = a, b, g

        # Assertions
        self.assertEqual(gmg.data, gpt.data.item())
        self.assertEqual(amg.grad, apt.grad.item())
        self.assertEqual(bmg.grad, bpt.grad.item())

if __name__ == '__main__':
    unittest.main()