import numpy as np

from network import KarateNetwork


def test_forward():
    k_net = KarateNetwork(3, 3, np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]]))
    loss = k_net.forward()
    assert isinstance(k_net, object)
    assert isinstance(loss, float)


def test_backward():
    k_net = KarateNetwork(3, 3, np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]]))
    k_net.forward()
    assert k_net.backward() is None
