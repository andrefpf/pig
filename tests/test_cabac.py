import numpy as np
from bitarray import bitarray

from jpig.entropy.cabac import Cabac


def test_specific_sequence():
    original = bitarray("1110 1101 1011 0111 1110 1111 1111 0111")
    expected_encoding = bitarray("1100 0111 1010 0101 1010 0000 11")

    cabac = Cabac()
    encoded = cabac.encode(original)
    decoded = cabac.decode(encoded, len(original))

    assert encoded == expected_encoding
    assert len(encoded) <= len(original)
    assert original == decoded

def test_small_sequence():
    original = bitarray("1110 1101 1011 0111")
    expected_encoding = bitarray("1100 0111 1010 0101")

    cabac = Cabac()
    encoded = cabac.encode(original)
    decoded = cabac.decode(encoded, len(original))

    assert encoded == expected_encoding
    assert len(encoded) <= len(original)
    assert original == decoded

def test_more_zeros_than_ones():
    # random sequence of booleans with more zeros than ones
    original = bitarray((np.random.random(100) < 0.2).tolist())

    cabac = Cabac()
    encoded = cabac.encode(original)
    decoded = cabac.decode(encoded, len(original))

    assert len(encoded) <= len(original)
    assert original == decoded


def test_more_ones_than_zeros():
    # random sequence of booleans with more ones than zeros
    original = bitarray((np.random.random(100) < 0.9).tolist())

    cabac = Cabac()
    encoded = cabac.encode(original)
    decoded = cabac.decode(encoded, len(original))

    assert len(encoded) <= len(original)
    assert original == decoded
