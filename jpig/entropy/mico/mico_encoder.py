import numpy as np
from bitarray import bitarray
from typing import Sequence

from jpig.entropy import CabacEncoder, FrequentistPM, ExponentialSmoothingPM
from jpig.metrics import RD
from jpig.utils.block_utils import split_blocks_in_half


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

        self.lower_bitplane = 0
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
        lower_bitplane: int = 0,
        upper_bitplane: int = 32,
    ) -> bitarray:
        self.clear()

        self.block = block
        self.lagrangian = lagrangian
        self.lower_bitplane = lower_bitplane
        self.upper_bitplane = upper_bitplane

        self.bitplane_sizes = self.calculate_bitplane_sizes()

    def apply_encoding(self, flags: Sequence[str], block_position: tuple[slice]):
        singular_value = all((slice.stop - slice.start) == 1 for slice in block_position)

        if flags == "Z":
            self.cabac.encode_bit(0, model=self.flags_model)

        elif flags == "C" and singular_value:
            pass 

        elif flags == "C" and not singular_value:
            pass

    @classmethod
    def calculate_bitplane_sizes(cls, block: np.ndarray):
        tmp_block = block.copy()
        bitplane_sizes = []

        for i in range(max(block.shape)):
            slices = tuple(slice(0, i) for _ in range(block.ndim))
            tmp_block[*slices] = 0

            bp = cls.find_max_bitplane(tmp_block)
            bitplane_sizes.append(bp)

        return bitplane_sizes

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()
