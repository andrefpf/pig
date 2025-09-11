from collections import deque

import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacEncoder
from jpig.metrics import RD
from jpig.utils.block_utils import bigger_possible_slice, split_shape_in_half

from .mico_optimizer import MicoOptimizer
from .mico_probability_handler import MicoProbabilityHandler

type Flags = deque[str]


ALL_ONES = np.iinfo(int).max


class MicoEncoder:
    """
    Multidimensional Image COdec - Encoder
    """

    def __init__(self):
        self.block: np.ndarray = np.array([], dtype=int)
        self.block_levels: np.ndarray = np.array([], dtype=int)
        self.level_bitplanes: np.ndarray = np.array([], dtype=int)

        self.lagrangian = 10_000
        self.lower_bitplane = 0

        self.flags: deque[str] = deque()
        self.estimated_rd = RD()

        self.prob_handler = MicoProbabilityHandler()
        self.bitstream = bitarray()
        self.cabac = CabacEncoder()

    def encode(self, block: np.ndarray, lagrangian: float = 10_000) -> bitarray:
        self.block = block
        self.lagrangian = lagrangian

        optimizer = MicoOptimizer(block, lagrangian)
        self.flags, self.estimated_rd = optimizer.optimize_tree()

        self.lower_bitplane = optimizer.lower_bitplane
        self.level_bitplanes = optimizer.level_bitplanes
        self.block_levels = optimizer.block_levels

        self.cabac.start(result=self.bitstream)
        self.apply_encoding(self.flags.copy(), bigger_possible_slice(block.shape))
        return self.cabac.end(fill_to_byte=True)

    def apply_encoding(self, flags: Flags, block_position: tuple[slice, ...]):
        sub_block = self.block[block_position]
        sub_levels = self.block_levels[block_position]
        upper_bitplanes = self.level_bitplanes[sub_levels]

        if np.all(upper_bitplanes < self.lower_bitplane) or np.all(upper_bitplanes <= 0):
            return

        if sub_block.size == 1:
            return self.encode_unit_block(block_position)

        flag = flags.popleft()
        model_split = self.prob_handler.split_model()
        model_block = self.prob_handler.block_model()

        if flag == "E":
            self.cabac.encode_bit(0, model=model_split)
            self.cabac.encode_bit(0, model=model_block)

        elif flag == "F":
            for upper_bitplane, value in zip(upper_bitplanes.flatten(), sub_block.flatten()):
                upper_bitplane = self.get_bitplane(block_position)
                self.encode_int(value, self.lower_bitplane, upper_bitplane, signed=True)

        elif flag == "S":
            self.cabac.encode_bit(1, model=model_split)
            for sub_pos in split_shape_in_half(block_position):
                self.apply_encoding(flags, sub_pos)

    def encode_unit_block(self, block_position: tuple[slice, ...]):
        value = self.block[block_position].flatten()[0]
        level = self.block_levels[block_position].flatten()[0]
        upper_bitplane = self.level_bitplanes[level]

        mask = ALL_ONES << self.lower_bitplane
        model = self.prob_handler.unit_model()
        self.cabac.encode_bit((value & mask) != 0, model=model)
        self.encode_int(value, self.lower_bitplane, upper_bitplane, signed=True)

    def encode_full(self, block_position: tuple[slice, ...]):
        pass

    def encode_int(
        self,
        value: int,
        lower_bitplane: int,
        upper_bitplane: int,
        signed: bool,
    ):
        absolute = np.abs(value)
        for i in range(lower_bitplane, upper_bitplane):
            bit = ((1 << i) & absolute) != 0
            model = self.prob_handler.int_model(i)
            self.cabac.encode_bit(bit, model=model)

        mask = (1 << lower_bitplane) - 1
        if signed and (absolute & ~mask) != 0:
            model = self.prob_handler.signal_model()
            self.cabac.encode_bit(value < 0, model=model)

    def get_bitplane(self, block_position: tuple[slice, ...]):
        level = self.get_level(block_position)
        return self.level_bitplanes[level]

    def get_level(self, block_position: tuple[slice, ...]):
        return max(s.start for s in block_position)
