import numpy as np
from jpig.entropy import MuleEncoder, MuleDecoder


def test_mule_easy():
    original = np.array(
        [
            [12, 8, 0, 2],
            [-7, 3, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 3, -2],
        ]
    )

    encoder = MuleEncoder()
    decoder = MuleDecoder()

    # Encoding with lagrangian equals to zero
    # i.e. without introducing losses
    max_bitplane = MuleEncoder.find_max_bitplane(original)
    encoded = encoder.encode(original, 0, lower_bitplane=0, upper_bitplane=max_bitplane)
    decoded = decoder.decode(encoded, original.shape, lower_bitplane=0, upper_bitplane=max_bitplane)

    assert np.allclose(original, decoded)


def test_mule_random():
    original = np.random.randint(0, 255, (9, 10, 8, 5, 2))

    # Encoding with lagrangian equals to zero
    # i.e. without introducing losses
    max_bitplane = MuleEncoder.find_max_bitplane(original)
    encoded = MuleEncoder().encode(original, 0, lower_bitplane=0, upper_bitplane=max_bitplane)
    decoded = MuleDecoder().decode(encoded, original.shape, lower_bitplane=0, upper_bitplane=max_bitplane)

    assert np.allclose(original, decoded)
