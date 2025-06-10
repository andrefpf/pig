import numpy as np
from jpig.entropy import MicoEncoder, MicoDecoder
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
    encoder = MicoEncoder()
    encoder.block = original
    assert encoder.calculate_bitplane_sizes() == [5, 4, 2, 2]


def test_mule_easy():
    original = np.array(
        [
            [12, 8, 0, 2],
            [-7, 3, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 3, -2],
        ]
    )

    encoder = MicoEncoder()
    decoder = MicoDecoder()


    # Encoding with lagrangian equals to zero
    # i.e. without introducing losses
    max_bitplane = MicoEncoder.find_max_bitplane(original)
    encoded = encoder.encode(original, 0, upper_bitplane=max_bitplane)
    decoded = decoder.decode(encoded, original.shape, upper_bitplane=max_bitplane)

    print(decoded)

    # assert encoder.flags == "SSLLSZLLS"
    # assert np.allclose(original, decoded)
