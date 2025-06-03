from jpig.entropy.cabac import Cabac
import numpy as np


def test_specific_sequence():
    # random sequence of booleans with more zeros than ones
    original = [
        True, True, True, False,
        True, True, False, True,
        True, False, True, True,
        False, True, True, True,
        True, True, True, False,
        True, True, True, True,
        True, True, True, True,
        False, True, True, True,
    ]  # fmt: skip

    expected_encoding = [
        True, True, False, False,
        False, True, True, True, 
        True, False, True, False, 
        False, True, False, True, 
        True, False, True, False, 
        False, False, False, False, 
        True, True
    ]  # fmt: skip


    cabac = Cabac()
    encoded = cabac.encode(original)
    decoded = cabac.decode(encoded, len(original))


    assert encoded == expected_encoding
    assert len(encoded) <= len(original)
    assert original == decoded



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
