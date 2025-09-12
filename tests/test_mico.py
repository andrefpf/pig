import numpy as np
import pytest

from jpig.entropy import MicoDecoder, MicoEncoder
from jpig.entropy.mico.mico_optimizer import MicoOptimizer


def test_mico_bitplanes():
    original = np.array(
        [
            [18, 8, 0, 2],
            [-7, 3, 0, 0],
            [0, 0, 1, -2],
            [0, 0, 3, -1],
        ]
    )  # fmt: skip
    assert list(MicoOptimizer.find_bitplane_per_level(original)) == [5, 4, 2, 2]


def test_mico_easy():
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
        lagrangian=1e-12,
    )
    decoded = decoder.decode(
        encoded,
        original.shape,
    )

    flags = "".join(encoder.flags)

    assert (flags == "SFSzvzzEF") or (flags == "SFFEF")
    assert np.allclose(original, decoded)
    assert encoder.lower_bitplane == decoder.lower_bitplane

    for e, d in zip(encoder.level_bitplanes, decoder.level_bitplanes):
        if d <= decoder.lower_bitplane:
            assert d > e
        else:
            assert d == e


# @pytest.mark.skip
def test_mico_random():
    original = np.random.randint(0, 255, (9, 10, 8, 5, 2))

    encoder = MicoEncoder()
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

    assert encoder.lower_bitplane == decoder.lower_bitplane
    assert np.allclose(original, decoded)

    for e, d in zip(encoder.level_bitplanes, decoder.level_bitplanes):
        if d <= decoder.lower_bitplane:
            assert d > e
        else:
            assert d == e
