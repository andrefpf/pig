import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacDecoder, FrequentistPM
from jpig.utils.block_utils import split_blocks_in_half


class MuleDecoder:
    def __init__(self):
        self.lower_bitplane = 0
        self.upper_bitplane = 32

        self.flags_probability_model = FrequentistPM()
        self.signals_probability_model = FrequentistPM()
        self.bitplane_probability_models = [FrequentistPM() for _ in range(32)]

        self.bitstream = bitarray()
        self.cabac = CabacDecoder()

    def decode(
        self,
        bitstream: bitarray,
        shape: tuple[int],
        *,
        lower_bitplane: int = 0,
        upper_bitplane: int = 32,
    ) -> np.ndarray:

        self.lower_bitplane = lower_bitplane
        self.upper_bitplane = upper_bitplane
        block = np.zeros(shape)

        self.cabac.start(bitstream)
        self.apply_decoding(block, self.upper_bitplane)
        self.cabac.end()
        return block

    def apply_decoding(self, block: np.ndarray, bitplane: int = 32):
        if block.size == 1:
            value = np.int64()

            for i in range(self.lower_bitplane, bitplane):
                bit = self.cabac.decode_bit(model=self.bitplane_probability_models[i])
                if bit:
                    value |= 1 << i

            signal = self.cabac.decode_bit(model=self.signals_probability_model)
            if signal:
                value = -value

            block[:] = value
            return

        flag = self._decode_flag()
        if flag == "Z":
            return

        elif flag == "L":
            return self.apply_decoding(block, bitplane - 1)

        elif flag == "S":
            for sub_block in split_blocks_in_half(block):
                self.apply_decoding(sub_block, bitplane)

        else:
            raise ValueError("Invalid encoding")

    def _decode_flag(self):
        first_bit = self.cabac.decode_bit(model=self.flags_probability_model)
        if first_bit:
            second_bit = self.cabac.decode_bit(model=self.flags_probability_model)
            if second_bit:  # 11
                return "S"
            else:  # 10
                return "L"
        else:  # 00
            return "Z"
