import numpy as np


def mse(baseline: np.ndarray, distorted: np.ndarray) -> float:
    baseline = np.array(baseline, dtype=np.float64)
    distorted = np.array(distorted, dtype=np.float64)
    return np.mean((baseline - distorted) ** 2)


def psnr(baseline: np.ndarray, distorted: np.ndarray, n_bits: int) -> float:
    max_value = (1 << n_bits) - 1
    _mse = mse(baseline, distorted)
    if _mse == 0:
        return np.inf
    return 10 * np.log10(max_value * max_value / _mse)


def energy(block: np.ndarray) -> float:
    return np.sum(block.astype(np.float64) ** 2, dtype=np.float64)
