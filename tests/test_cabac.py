import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacEncoder, CabacDecoder, ProbabilityModel


def test_specific_sequence():
    original = bitarray("1110 1101 1011 0111 1110 1111 1111 0111")
    expected_encoding = bitarray("11000001011010010111100011")

    encoded = CabacEncoder().encode(original)
    decoded = CabacDecoder().decode(encoded, len(original))

    assert encoded == expected_encoding
    assert len(encoded) <= len(original)
    assert original == decoded


def test_more_zeros_than_ones():
    # random sequence of booleans with more zeros than ones
    original = bitarray((np.random.random(100) < 0.2).tolist())

    encoded = CabacEncoder().encode(original)
    decoded = CabacDecoder().decode(encoded, len(original))

    assert len(encoded) <= len(original)
    assert original == decoded


def test_more_ones_than_zeros():
    # random sequence of booleans with more ones than zeros
    original = bitarray((np.random.random(100) < 0.9).tolist())

    encoded = CabacEncoder().encode(original)
    decoded = CabacDecoder().decode(encoded, len(original))

    assert len(encoded) <= len(original)
    assert original == decoded


def test_mixed_models():
    # The data
    part_1 = (np.random.random(100) < 0.3).tolist()
    part_2 = (np.random.random(80) < 0.6).tolist()

    # Encode the data
    model_1_encoder = ProbabilityModel()
    model_2_encoder = ProbabilityModel()
    encoded_bitstream = bitarray()
    encoder = CabacEncoder().start(encoded_bitstream)

    encoder.use_model(model_1_encoder)
    for i in part_1:
        encoder.encode_bit(i)

    encoder.use_model(model_2_encoder)
    for i in part_2:
        encoder.encode_bit(i)

    encoder.end()

    # Decode the data
    model_1_decoder = ProbabilityModel()
    model_2_decoder = ProbabilityModel()
    decoded_bitstream = bitarray()
    decoder = CabacDecoder().start(encoded_bitstream, decoded_bitstream)

    decoder.use_model(model_1_decoder)
    for _ in range(len(part_1)):
        decoder.decode_bit()

    decoder.use_model(model_2_decoder)
    for _ in range(len(part_2)):
        decoder.decode_bit()

    decoder.end()

    assert bitarray(part_1 + part_2) == decoded_bitstream
