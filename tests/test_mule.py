from collections import deque

import numpy as np

from jpig.entropy import MuleDecoder, MuleEncoder


def test_mule_easy():
    original = np.array(
        [
            [18, 8, 0,  2],
            [-7, 3, 0,  0],
            [ 0, 0, 1,  1],
            [ 0, 0, 3, -2],
        ]
    )  # fmt: skip

    encoder = MuleEncoder()
    decoder = MuleDecoder()

    # Encoding with lagrangian equals to zero
    # i.e. without introducing losses
    encoded = encoder.encode(
        original,
        0,
    )
    decoded = decoder.decode(
        encoded,
        original.shape,
        upper_bitplane=encoder.upper_bitplane,
    )

    assert encoder.flags == deque("SSLLLSZLLLS")
    assert np.allclose(original, decoded)


def test_mule_random():
    original = np.random.randint(0, 255, (9, 10, 8, 5, 2))

    # Encoding with lagrangian equals to zero
    # i.e. without introducing losses
    encoder = MuleEncoder()
    decoder = MuleDecoder()

    encoded = encoder.encode(
        original,
        0,
    )
    decoded = decoder.decode(
        encoded,
        original.shape,
        upper_bitplane=encoder.upper_bitplane,
    )

    assert np.allclose(original, decoded)
