from jpig.utils.block_utils import split_shape_in_half


def test_shape_split():
    shape = (32, 16)
    expected_slices = [
        (slice(0, 16), slice(0, 8)),
        (slice(0, 16), slice(8, 16)),
        (slice(16, 32), slice(0, 8)),
        (slice(16, 32), slice(8, 16)),
    ]

    for expected, slices in zip(expected_slices, split_shape_in_half(shape)):
        assert expected == slices


def test_slices_split():
    slices = (slice(16, 32), slice(8, 16))
    expected_subslices = [
        (slice(16, 24), slice(8, 12)),
        (slice(16, 24), slice(12, 16)),
        (slice(24, 32), slice(8, 12)),
        (slice(24, 32), slice(12, 16)),
    ]

    for expected, sub_slices in zip(expected_subslices, split_shape_in_half(slices)):
        assert expected == sub_slices
