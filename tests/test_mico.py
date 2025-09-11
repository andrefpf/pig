import numpy as np
import pytest

from jpig.entropy import MicoDecoder, MicoEncoderOld


def test_mico_bitplanes():
    original = np.array(
        [
            [18, 8, 0, 2],
            [-7, 3, 0, 0],
            [0, 0, 1, -2],
            [0, 0, 3, -1],
        ]
    )  # fmt: skip
    encoder = MicoEncoderOld()
    encoder.block = original
    assert encoder._calculate_bitplane_sizes() == [5, 4, 2, 2]


def test_mico_easy():
    original = np.array(
        [
            [18, 8, 0,  2],
            [-7, 3, 0,  0],
            [ 0, 0, 1,  1],
            [ 0, 0, 3, -2],
        ]
    )  # fmt: skip

    encoder = MicoEncoderOld()
    decoder = MicoDecoder()

    # Encoding with lagrangian equals to zero
    # i.e. without introducing losses
    encoded = encoder.encode(
        original,
        lagrangian=1e-6,
    )
    decoded = decoder.decode(
        encoded,
        original.shape,
    )

    assert encoder.flags == "SFSzvzzEF"
    assert encoder.bitplane_sizes == decoder.bitplane_sizes
    assert np.allclose(original, decoded)


# @pytest.mark.skip
def test_mico_random():
    original = np.random.randint(0, 255, (9, 10, 8, 5, 2))

    encoder = MicoEncoderOld()
    decoder = MicoDecoder()

    # Encoding with lagrangian equals to zero
    # i.e. without introducing losses
    encoded = encoder.encode(
        original,
        lagrangian=0,
    )
    decoded = decoder.decode(
        encoded,
        original.shape,
    )

    assert encoder.bitplane_sizes == decoder.bitplane_sizes
    assert np.allclose(original, decoded)
