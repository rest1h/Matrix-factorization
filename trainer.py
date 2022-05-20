from typing import List

import numpy as np

from network import KarateNetwork


class Trainer:
    def __init__(
        self, dim: int, n_node: int, adj: np.ndarray, epoch: int, lr: float = 0.01
    ) -> None:
        self.net = KarateNetwork(dim, n_node, adj, lr)
        self.epoch = epoch

    def train(self) -> List[float]:
        result = []
        for epoch in range(self.epoch):
            result.append(self._train(epoch))

        return result

    def _train(self, epoch) -> float:
        loss = self.net.forward()
        print(f"Epoch: {epoch}, Loss: {loss}")
        self.net.backward()

        return loss
