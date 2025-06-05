import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacEncoder, CabacDecoder, ProbabilityModel


def test_specific_sequence():
    original = bitarray("1110 1101 1011 0111 1110 1111 1111 0111")
    expected_encoding = bitarray("1100 0001 0110 1001 0111 1000 11")

    encoded = CabacEncoder().encode(original)
    decoded = CabacDecoder().decode(encoded, len(original))

    assert encoded == expected_encoding
    assert len(encoded) <= len(original)
    assert original == decoded

def test_specific_sequence_filling_to_byte():
    original = bitarray("1110 1101 1011 0111 1110 1111 1111 0111")
    expected_encoding = bitarray("0000 0011 0000 0101 1010 0101 1110 0011")

    encoded = CabacEncoder().encode(original, fill_to_byte=True)
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
    part_2 = (np.random.random(20) < 0.6).tolist()
    part_3 = (np.random.random(80) < 0.8).tolist()

    encoder_models = [
        ProbabilityModel(),
        ProbabilityModel(),
        ProbabilityModel(),
    ]

    decoder_models = [
        ProbabilityModel(),
        ProbabilityModel(),
        ProbabilityModel(),
    ]

    # Encode the data
    encoder = CabacEncoder().start()

    for i in part_1:
        encoder.encode_bit(i, model=encoder_models[0])

    for i in part_2:
        encoder.encode_bit(i, model=encoder_models[1])

    for i in part_3:
        encoder.encode_bit(i, model=encoder_models[2])

    encoded_bitstream = encoder.end()

    # Decode the data
    decoder = CabacDecoder().start(encoded_bitstream)

    for _ in range(len(part_1)):
        decoder.decode_bit(model=decoder_models[0])

    for _ in range(len(part_2)):
        decoder.decode_bit(model=decoder_models[1])

    for _ in range(len(part_3)):
        decoder.decode_bit(model=decoder_models[2])

    decoded_bitstream = decoder.end()

    assert bitarray(part_1 + part_2 + part_3) == decoded_bitstream
    for e, d in zip(encoder_models, decoder_models):
        assert e == d
