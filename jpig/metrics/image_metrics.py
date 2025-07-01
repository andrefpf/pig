import numpy as np


def mse(baseline: np.ndarray, distorted: np.ndarray) -> float:
    return np.mean((baseline - distorted) ** 2)


def psnr(baseline: np.ndarray, distorted: np.ndarray, n_bits: int) -> float:
    max_value = (1 << n_bits) - 1
    return 10 * np.log10(max_value * max_value / mse(baseline, distorted))


def energy(block: np.ndarray) -> float:
    return np.sum(block**2)
