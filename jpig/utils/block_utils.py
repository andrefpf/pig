import numpy as np
from itertools import product, pairwise


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


def split_blocks_equal_size(block: np.ndarray, block_size: int) -> list[np.ndarray]:
    slices_per_dimension = []
    for size in block.shape:
        slices_current_dimension = tuple(
            slice(a, b) for (a, b) in
            pairwise(range(0, size + 1, block_size))
        )  # fmt: skip
        slices_per_dimension.append(tuple(slices_current_dimension))

    split_blocks = []
    for slices in product(*slices_per_dimension):
        split_block = block[*slices]
        split_blocks.append(split_block)

    return split_blocks
