import numpy as np

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
    max_bitplane = MicoEncoder.find_max_bitplane(original)
    encoded = encoder.encode(
        original,
        0,
        upper_bitplane=max_bitplane,
    )
    decoder.bitplane_sizes = encoder.bitplane_sizes
    decoded = decoder.decode(
        encoded,
        original.shape,
        upper_bitplane=max_bitplane,
    )

    assert encoder.flags == "CCCCCCCZCZZZCCCCC"
    assert np.allclose(original, decoded)


def test_mule_random():
    original = np.random.randint(0, 255, (9, 10, 8, 5, 2))

    encoder = MicoEncoder()
    decoder = MicoDecoder()

    # Encoding with lagrangian equals to zero
    # i.e. without introducing losses
    max_bitplane = MicoEncoder.find_max_bitplane(original)
    encoded = encoder.encode(
        original,
        0,
        upper_bitplane=max_bitplane,
    )
    decoder.bitplane_sizes = encoder.bitplane_sizes
    decoded = decoder.decode(
        encoded,
        original.shape,
        upper_bitplane=max_bitplane,
    )

    assert np.allclose(original, decoded)
