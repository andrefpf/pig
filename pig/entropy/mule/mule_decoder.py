import numpy as np
from bitarray import bitarray

from pig.entropy import CabacDecoder, FrequentistPM
from pig.utils.block_utils import split_blocks_in_half

from .mule_probability_handler import MuleProbabilityHandler


class MuleDecoder:
    def __init__(self):
        self.lower_bitplane = 0
        self.upper_bitplane = 32

        self.prob_handler = MuleProbabilityHandler()
        self.bitstream = bitarray()
        self.cabac = CabacDecoder()

    def decode(
        self,
        bitstream: bitarray,
        shape: tuple[int],
        *,
        upper_bitplane: int,
    ) -> np.ndarray:
        self.upper_bitplane = upper_bitplane
        block = np.zeros(shape, dtype=np.int32)

        self.cabac.start(bitstream)
        self.lower_bitplane = self.decode_int(0, 5, signed=False)
        self.apply_decoding(block, self.upper_bitplane)
        self.cabac.end()
        return block

    def apply_decoding(self, block: np.ndarray, bitplane: int = 32):
        if bitplane < self.lower_bitplane or bitplane <= 0:
            return

        if block.size == 1:
            block[:] = self.decode_int(self.lower_bitplane, bitplane, signed=True)
            return

        flag = self._decode_flag(bitplane)
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
        *,
        signed: bool,
    ) -> int:
        value = 0

        for i in range(lower_bitplane, upper_bitplane):
            bit = self.cabac.decode_bit(model=self.prob_handler.int_model(i))
            value |= bit << i

        if signed and value != 0:
            signal = self.cabac.decode_bit(model=self.prob_handler.signal_model())
            if signal:
                value = -value

        return value

    def _decode_flag(self, bitplane: int):
        model_0 = self.prob_handler.flag_model(bitplane, 0)
        model_1 = self.prob_handler.flag_model(bitplane, 1)
        first_bit = self.cabac.decode_bit(model=model_0)

        if first_bit:
            return "Z"  # 1
        else:
            second_bit = self.cabac.decode_bit(model=model_1)

            if second_bit:
                return "S"  # 01
            else:
                return "L"  # 00
