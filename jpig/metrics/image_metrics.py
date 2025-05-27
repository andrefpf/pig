import numpy as np


def mse(baseline: np.ndarray, modified: np.ndarray) -> float:
    return np.sum((baseline - modified) ** 2)


def psnr(baseline: np.ndarray, modified: np.ndarray, n_bits: int) -> float:
    max_value = (1 << n_bits) - 1
    return 10 * np.log10(max_value * max_value / mse(baseline, modified))
