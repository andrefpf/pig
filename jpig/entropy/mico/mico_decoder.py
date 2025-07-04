import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacDecoder, FrequentistPM
from jpig.utils.block_utils import bigger_possible_slice, split_shape_in_half


class MicoDecoder:
    """
    Multidimensional Image COdec - Decoder
    """

    def __init__(self):
        self.upper_bitplane = 32
        self.bitplane_sizes = []

        self.split_flags_model = FrequentistPM()
        self.unit_flags_model = FrequentistPM()
        self.block_flags_model = FrequentistPM()

        self.signals_model = FrequentistPM()
        self.bitplane_models = [FrequentistPM() for _ in range(32)]
        self.bitplane_sizes_model = FrequentistPM()

        self.bitstream = bitarray()
        self.cabac = CabacDecoder()

    def decode(
        self,
        bitstream: bitarray,
        shape: tuple[int],
        *,
        upper_bitplane: int = 32,
    ) -> np.ndarray:
        self.upper_bitplane = upper_bitplane
        self.block = np.zeros(shape, dtype=np.int32)

        self.cabac.start(bitstream)
        self.decode_bitplane_sizes()
        self.apply_decoding(bigger_possible_slice(shape))
        self.cabac.end()
        return self.block

    def apply_decoding(self, block_position: tuple[slice]):
        sub_block = self.block[block_position]
        flag = self._decode_flag(sub_block.size == 1)

        if flag == "S":  # Split
            for sub_pos in split_shape_in_half(block_position):
                self.apply_decoding(sub_pos)

        elif flag == "F":  # Full
            dims = np.mgrid[block_position]
            levels: np.ndarray = np.max(dims, axis=0)
            for i, level in np.ndenumerate(levels):
                upper_bitplane = self.bitplane_sizes[level]
                sub_block[i] = self.decode_value(upper_bitplane)

        elif flag == "E":  # Empty
            assert sub_block.size != 1

        elif flag == "v":  # Value
            assert sub_block.size == 1
            bitplane = self._get_bitplane(block_position)
            self.block[block_position] = self.decode_value(bitplane)

        elif flag == "z":  # Zero
            assert sub_block.size == 1

        else:
            raise ValueError(f'Invalid encoding flag "{flag}"')

    def decode_bitplane_sizes(self):
        self.bitplane_sizes = []
        counter = 0
        for _ in range(max(self.block.shape)):
            while self.cabac.decode_bit(model=self.bitplane_sizes_model):
                counter += 1
            self.bitplane_sizes.append(counter)
        self.bitplane_sizes.reverse()

    def decode_value(self, upper_bitplane: int = 32) -> int:
        value = 0
        for i in range(0, upper_bitplane):
            bit = self.cabac.decode_bit(model=self.bitplane_models[i])
            if bit:
                value |= 1 << i

        signal = self.cabac.decode_bit(model=self.signals_model)
        if signal:
            value = -value

        return value

    def _decode_flag(self, unitary: bool):
        if unitary:
            unit_flag = self.cabac.decode_bit(model=self.unit_flags_model)
            if unit_flag:
                return "v"
            else:
                return "z"

        else:
            split_flag = self.cabac.decode_bit(model=self.split_flags_model)
            if split_flag:
                return "S"
            else:
                block_flag = self.cabac.decode_bit(model=self.block_flags_model)
                if block_flag:
                    return "F"
                else:
                    return "E"

    def _get_bitplane(self, block_position: tuple[slice]):
        level = max(s.start for s in block_position)
        return self.bitplane_sizes[level]
