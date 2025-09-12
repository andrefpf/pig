from collections import deque

import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacEncoder
from jpig.metrics import RD
from jpig.utils.block_utils import bigger_possible_slice, split_shape_in_half

from .mico_optimizer import MicoOptimizer
from .mico_probability_handler import MicoProbabilityHandler

type Flags = deque[str]


ALL_ONES = np.iinfo(np.int32).max


class MicoEncoder:
    """
    Multidimensional Image COdec - Encoder
    """

    def __init__(self):
        self.block: np.ndarray = np.array([], dtype=np.int32)
        self.block_levels: np.ndarray = np.array([], dtype=np.int32)
        self.level_bitplanes: np.ndarray = np.array([], dtype=np.int32)

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
        self.encode_bitplane_sizes()
        self.apply_encoding(self.flags.copy(), bigger_possible_slice(block.shape))
        return self.cabac.end(fill_to_byte=True)

    def encode_bitplane_sizes(self):
        last_size = self.lower_bitplane
        self.encode_int(self.lower_bitplane, 0, 5, signed=False)
        for size in reversed(self.level_bitplanes):
            delta = max(size - last_size, 0)
            for _ in range(delta):
                self.cabac.encode_bit(1, model=self.prob_handler.bitplanes_model())
            self.cabac.encode_bit(0, model=self.prob_handler.bitplanes_model())
            last_size += delta
        self.prob_handler.clear()

    def apply_encoding(self, flags: Flags, block_position: tuple[slice, ...]):
        sub_block = self.block[block_position]
        sub_levels = self.block_levels[block_position]
        upper_bitplanes = self.level_bitplanes[sub_levels]

        if np.all(upper_bitplanes <= self.lower_bitplane):
            return

        if np.all(upper_bitplanes <= 0):
            return

        flag = flags.popleft()
        max_bp = self.get_bitplane(block_position)
        model_split = self.prob_handler.split_model(max_bp)
        model_block = self.prob_handler.block_model(max_bp)

        if flag == "S":
            self.cabac.encode_bit(1, model=model_split)
            for sub_pos in split_shape_in_half(block_position):
                self.apply_encoding(flags, sub_pos)

        elif flag == "E":
            self.cabac.encode_bit(0, model=model_split)
            self.cabac.encode_bit(0, model=model_block)
            return

        elif flag == "F":
            self.cabac.encode_bit(0, model=model_split)
            self.cabac.encode_bit(1, model=model_block)
            for i, upper_bitplane in np.ndenumerate(upper_bitplanes):
                value = sub_block[i].flatten()[0]
                self.encode_int(value, self.lower_bitplane, upper_bitplane, signed=True)

        elif flag == "z":
            assert sub_block.size == 1
            model = self.prob_handler.unit_model()
            self.cabac.encode_bit(0, model=model)
            return

        elif flag == "v":
            assert sub_block.size == 1
            value = sub_block.flatten()[0]
            upper_bitplane = upper_bitplanes.flatten()[0]

            model = self.prob_handler.unit_model()
            self.cabac.encode_bit(1, model=model)
            self.encode_int(value, self.lower_bitplane, upper_bitplane, signed=True)

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
