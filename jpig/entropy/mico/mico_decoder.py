import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacDecoder, FrequentistPM
from jpig.utils.block_utils import split_shape_in_half, bigger_possible_slice


class MicoDecoder:
    """
    Multidimensional Image COdec - Decoder
    """

    def __init__(self):
        self.upper_bitplane = 32
        self.bitplane_sizes = []

        self.flags_model = FrequentistPM()
        self.signals_model = FrequentistPM()
        self.bitplane_models = [FrequentistPM() for _ in range(32)]

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
        self.apply_decoding(bigger_possible_slice(shape))
        self.cabac.end()
        return self.block

    def apply_decoding(self, block_position: tuple[slice]):
        flag = self._decode_flag()

        if flag == "Z":
            return

        sub_block = self.block[block_position]
        if sub_block.size > 1:
            for sub_pos in split_shape_in_half(block_position):
                self.apply_decoding(sub_pos)
            return

        bitplane = self._get_bitplane(block_position)
        value = 0
        for i in range(0, bitplane):
            bit = self.cabac.decode_bit(model=self.bitplane_models[i])
            if bit:
                value |= 1 << i

        signal = self.cabac.decode_bit(model=self.signals_model)
        if signal:
            value = -value

        self.block[block_position] = value

    def _decode_flag(self):
        flag = self.cabac.decode_bit(model=self.flags_model)
        if flag:
            return "C"
        else:
            return "Z"

    def _get_bitplane(self, block_position: tuple[slice]):
        level = max(s.start for s in block_position)
        return self.bitplane_sizes[level]
