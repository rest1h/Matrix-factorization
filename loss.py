import numpy as np


def factorize_matrix(
    z_emb_u: np.ndarray, z_emb_v: np.ndarray, adj: np.ndarray
) -> np.ndarray:
    e_diff = np.dot(z_emb_u.T, z_emb_v) - adj
    return np.sum(np.square(e_diff)), e_diff
