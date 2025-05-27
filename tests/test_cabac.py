from jpig.entropy.cabac import Cabac
import numpy as np


def test_more_zeros_than_ones():
    # random sequence of booleans with more zeros than ones
    original = list(np.random.random(100) < 0.2)

    cabac = Cabac()
    encoded = cabac.encode(original)
    decoded = cabac.decode(encoded, len(original))

    assert len(encoded) <= len(original)
    assert original == decoded


def test_more_ones_than_zeros():
    # random sequence of booleans with more ones than zeros
    original = list(np.random.random(100) < 0.9)

    cabac = Cabac()
    encoded = cabac.encode(original)
    decoded = cabac.decode(encoded, len(original))

    assert len(encoded) <= len(original)
    assert original == decoded
