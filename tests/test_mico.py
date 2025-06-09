import numpy as np
from jpig.entropy import MicoEncoder
from jpig.utils.block_utils import split_shape_in_half


def test_mico_bitplanes():
    original = np.array(
        [
            [18, 8, 0, 2],
            [-7, 3, 0, 0],
            [0, 0, 1, -2],
            [0, 0, 3, -1],
        ]
    )

    assert MicoEncoder.calculate_bitplane_sizes(original) == [5, 4, 2, 2]
