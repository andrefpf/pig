import numpy as np


def max_level(block_position: tuple[slice, ...] | tuple[int, ...]):
    stop_position = []
    for s in block_position:
        stop = s.stop if isinstance(s, slice) else s
        stop_position.append(stop)
    return max(stop_position)


def get_level(block_position: tuple[slice, ...] | tuple[int, ...]):
    start_position = []
    for s in block_position:
        start = s.start if isinstance(s, slice) else s
        start_position.append(start)
    return max(start_position)


def get_shape_levels(shape: tuple[int, ...]) -> np.ndarray:
    """
    The levels of a (4, 5) block are organized as follows:

    0, 1, 2, 3, 4
    1, 1, 2, 3, 4
    2, 2, 2, 3, 4
    3, 3, 3, 3, 4
    """

    blocks_level = np.zeros(shape, dtype=np.int32)
    total_levels = max_level(shape)
    for position in np.ndindex(*shape):
        blocks_level[position] = get_level(position)
    return blocks_level.clip(0, total_levels - 1)


def get_block_levels(block: np.ndarray) -> np.ndarray:
    return get_shape_levels(block.shape)
