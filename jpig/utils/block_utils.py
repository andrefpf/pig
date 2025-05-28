import numpy as np
from itertools import product


def split_blocks_in_half(block: np.ndarray) -> list[np.ndarray]:
    slices_per_dimension = []
    for size in block.shape:
        half = size // 2

        if half == 0:
            slices = (slice(0, size),)
        else:
            slices = (slice(0, half), slice(half, size))

        slices_per_dimension.append(slices)

    split_blocks = []
    for slices in product(*slices_per_dimension):
        split_block = block[*slices]
        split_blocks.append(split_block)

    return split_blocks
