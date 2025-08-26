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
        upper_bitplane: int = 32,
    ) -> np.ndarray:
        self.upper_bitplane = upper_bitplane
        block = np.zeros(shape)

        self.cabac.start(bitstream)
        self.lower_bitplane = self.decode_int(0, 5, signed=False)
        self.apply_decoding(block, self.upper_bitplane)
        self.cabac.end()
        return block

    def apply_decoding(self, block: np.ndarray, bitplane: int = 32):
        if block.size == 1:
            block[:] = self.decode_int(self.lower_bitplane, bitplane)
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

    def decode_int(
        self,
        lower_bitplane: int,
        upper_bitplane: int,
        signed: bool = True,
    ) -> int:
        value = 0

        for i in range(lower_bitplane, upper_bitplane):
            bit = self.cabac.decode_bit(model=self.bitplane_probability_models[i])
            value |= bit << i

        if signed:
            signal = self.cabac.decode_bit(model=self.signals_probability_model)
            if signal:
                value = -value

        return value

    def _decode_flag(self):
        first_bit = self.cabac.decode_bit(model=self.flags_probability_model)
        if first_bit:
            return "Z"  # 1
        else:
            second_bit = self.cabac.decode_bit(model=self.flags_probability_model)
            if second_bit:  # 01
                return "S"
            else:  # 00
                return "L"
