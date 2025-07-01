import numpy as np
import pytest

from jpig.entropy import MicoDecoder, MicoEncoder


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
    assert encoder._calculate_bitplane_sizes() == [5, 4, 2, 2]


def test_mule_easy():
    original = np.array(
        [
            [18, 8, 0,  2],
            [-7, 3, 0,  0],
            [ 0, 0, 1,  1],
            [ 0, 0, 3, -2],
        ]
    )  # fmt: skip

    encoder = MicoEncoder()
    decoder = MicoDecoder()

    # Encoding with lagrangian equals to zero
    # i.e. without introducing losses
    encoded = encoder.encode(
        original,
        lagrangian=1,
    )
    # decoded = decoder.decode(
    #     encoded,
    #     original.shape,
    # )

    print(len(encoded), encoder.estimated_rd, encoder.flags)

    # assert encoder.flags == "SASZA"
    # assert encoder.bitplane_sizes == decoder.bitplane_sizes
    # assert np.allclose(original, decoded)


def test_mule_random():
    return
    original = np.random.randint(0, 255, (9, 10, 8, 5, 2))

    encoder = MicoEncoder()
    decoder = MicoDecoder()

    # Encoding with lagrangian equals to zero
    # i.e. without introducing losses
    encoded = encoder.encode(
        original,
        lagrangian=0,
    )
    # decoded = decoder.decode(
    #     encoded,
    #     original.shape,
    # )

    # assert encoder.bitplane_sizes == decoder.bitplane_sizes
    # assert np.allclose(original, decoded)
