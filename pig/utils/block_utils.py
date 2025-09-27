from itertools import chain, pairwise, product
from typing import Generator

import numpy as np


def split_shape_in_half(_slices: tuple[int] | tuple[slice]) -> Generator[slice, None, None]:
    if len(_slices) == 0:
        raise StopIteration()

    if isinstance(_slices[0], int):
        _slices = bigger_possible_slice(_slices)

    slices_per_dimension = []
    for _slice in _slices:
        half = _slice.start + (_slice.stop - _slice.start) // 2

        if half == 0:
            slices = (slice(_slice.start, _slice.stop),)
        else:
            slices = (slice(_slice.start, half), slice(half, _slice.stop))

        slices_per_dimension.append(slices)

    for slices in product(*slices_per_dimension):
        yield slices


def split_blocks_in_half(block: np.ndarray) -> Generator[np.ndarray, None, None]:
    for slices in split_shape_in_half(block.shape):
        yield block[slices]


def split_blocks_equal_size(block: np.ndarray, block_size: int) -> list[np.ndarray]:
    slices_per_dimension = []
    for size in block.shape:
        keypoints = chain(range(0, size, block_size), [size])
        slices_current_dimension = tuple(
            slice(a, b) for (a, b) in
            pairwise(keypoints)
        )  # fmt: skip
        slices_per_dimension.append(tuple(slices_current_dimension))

    split_blocks = []
    for slices in product(*slices_per_dimension):
        split_block = block[slices]
        split_blocks.append(split_block)

    return split_blocks


def bigger_possible_slice(shape: tuple[int]) -> tuple[slice]:
    return tuple(slice(0, size) for size in shape)
