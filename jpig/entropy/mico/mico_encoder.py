import numpy as np
from bitarray import bitarray
from typing import Sequence

from jpig.entropy import CabacEncoder, FrequentistPM, ExponentialSmoothingPM
from jpig.metrics import RD
from jpig.utils.block_utils import split_shape_in_half, bigger_possible_slice


class MicoEncoder:
    """
    Multidimensional Image COdec - Encoder
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.estimated_rd = RD()
        self.flags = ""
        self.block = np.array([])
        self.bitplane_sizes = []

        self.upper_bitplane = 32
        self.lagrangian = 10_000

        self.flags_model = ExponentialSmoothingPM()
        self.bitplane_sizes_model = ExponentialSmoothingPM()
        self.signals_probability_model = ExponentialSmoothingPM()
        self.bitplane_probability_models = [ExponentialSmoothingPM() for _ in range(32)]

        self.bitstream = bitarray()
        self.cabac = CabacEncoder()

    def encode(
        self,
        block: np.ndarray,
        lagrangian: float = 10_000,
        *,
        upper_bitplane: int = 32,
    ) -> bitarray:
        self.clear()

        self.block = block
        self.lagrangian = lagrangian
        self.upper_bitplane = upper_bitplane

        self.bitplane_sizes = self.calculate_bitplane_sizes()
        # self.encode_bitplane_sizes()

        self.cabac.start(result=self.bitstream)
        self.apply_encoding(list("CCCCCCCZCZZZCCCCC"), bigger_possible_slice(block.shape))
        return self.cabac.end(fill_to_byte=True)

    def apply_encoding(self, flags: list[str], block_position: tuple[slice]):
        flag = flags.pop(0)
        if flag not in ["Z", "C"]:
            raise ValueError("Invalid encoding")

        if flag == "Z":
            self.cabac.encode_bit(0, model=self.flags_model)
            return

        self.cabac.encode_bit(1, model=self.flags_model)
        sub_block = self.block[block_position]
        if sub_block.size > 1:
            for sub_pos in split_shape_in_half(sub_block.shape):
                self.apply_encoding(flags, sub_pos)
            return

        bitplane = self._get_bitplane(block_position)
        value = sub_block.flatten()[0]
        for i in range(0, bitplane):
            bit = (1 << i) & np.abs(value) != 0
            self.cabac.encode_bit(bit, model=self.bitplane_probability_models[i])
        self.cabac.encode_bit(value < 0, model=self.signals_probability_model)

    def encode_bitplane_sizes(self):
        last_size = 0
        for size in reversed(self.bitplane_sizes):
            difference = size - last_size
            for i in range(difference):
                self.cabac.encode_bit(1, model=self.bitplane_sizes_model)
            self.cabac.encode_bit(0, model=self.bitplane_sizes_model)
            last_size = size

    def _get_bitplane(self, block_position: tuple[slice]):
        level = max(s.stop for s in block_position)
        return self.bitplane_sizes[level]

    def calculate_bitplane_sizes(self):
        tmp_block = self.block.copy()
        bitplane_sizes = []

        for i in range(max(self.block.shape)):
            slices = tuple(slice(0, i) for _ in range(self.block.ndim))
            tmp_block[*slices] = 0

            bp = self.find_max_bitplane(tmp_block)
            bitplane_sizes.append(bp)

        return bitplane_sizes

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()
