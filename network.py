import numpy as np

from loss import factorize_matrix


class KarateNetwork:
    def __init__(self, dim: int, n_node: int, adj: np.ndarray, lr: float = 0.01):
        self._node_emb_u = np.random.random((dim, n_node))
        self._node_emb_v = np.random.random((dim, n_node))
        self.adj = adj
        self.lr = lr
        self.e_diff = None

    def forward(self):
        loss, self.e_diff = factorize_matrix(
            self._node_emb_u, self._node_emb_v, self.adj
        )
        return loss

    def backward(self):
        self._node_emb_u -= self.lr * np.dot(self.e_diff, self._node_emb_v.T).T
        self._node_emb_v -= self.lr * np.dot(self.e_diff, self._node_emb_u.T).T

    @property
    def node_emb_u(self):
        return self._node_emb_u

    @node_emb_u.setter
    def node_emb_u(self, value):
        self._node_emb_u = value

    @property
    def node_emb_v(self):
        return self._node_emb_v

    @node_emb_v.setter
    def node_emb_v(self, value):
        self._node_emb_v = value
