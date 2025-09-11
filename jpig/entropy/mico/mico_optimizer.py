from collections import deque
from typing import Sequence

import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacEncoder, FrequentistPM
from jpig.metrics import RD, energy
from jpig.utils.block_utils import split_blocks_in_half

from .mico_probability_handler import MicoProbabilityHandler

type Flags = deque[str]


class MicoOptimizer:
    def __init__(self, block: np.ndarray, lagrangian: int = 10_000):
        self.block = block
        self.lagrangian = lagrangian
        self.prob_handler = MicoProbabilityHandler()

        self._block_levels = MicoOptimizer.get_block_levels(self.block)
        self._level_bitplane = MicoOptimizer.find_bitplane_per_level(self.block)

    def optimize_lower_bitplane(self) -> int:
        lower_bitplane = 0
        accumulated_rate = 0
        best_cost = float("inf")

        magnitudes = np.abs(self.block).flatten()
        upper_bp = max(self._level_bitplane)

        for i in reversed(range(0, upper_bp)):
            bit_position = 1 << i
            lower_mask = (1 << i) - 1
            non_zeroed = self._block_levels.flatten() >= i

            model = self.prob_handler.int_model(i)
            bits_to_encode = magnitudes[non_zeroed] & bit_position != 0
            for bit in bits_to_encode:
                accumulated_rate += model.add_and_estimate_bit(bit)

            sign_rate = np.sum(non_zeroed)
            total_rate = accumulated_rate + sign_rate
            rd = RD(total_rate, energy(magnitudes & lower_mask))

            if rd.cost(self.lagrangian) < best_cost:
                best_cost = rd.cost(self.lagrangian)
                lower_bitplane = i

        self.prob_handler.clear()
        return lower_bitplane

    def optimize_tree(
        self,
        block: np.ndarray,
        lower_bp: int,
    ) -> tuple[Flags, RD]:
        pass

    def _estimate_full(
        self,
        block_position: tuple[slice, ...],
        lower_bp: int,
    ) -> tuple[Flags, RD]:
        flags = deque("F")
        sub_block = self.block[block_position]
        sub_levels = self._block_levels[block_position]

        rd = RD()
        rd.rate += self.prob_handler.split_model().add_and_estimate_bit(0)
        rd.rate += self.prob_handler.block_model().add_and_estimate_bit(0)

        for level, value in zip(sub_levels.flatten(), sub_block.flatten()):
            upper_bp = self._level_bitplane[level]
            rd += self._estimate_integer(value, lower_bp, upper_bp, signed=True)

        return flags, rd

    def _estimate_empty(self, block_position: tuple[slice, ...]) -> tuple[Flags, RD]:
        flags = deque("E")
        sub_block = self.block[block_position]

        rd = RD()
        rd.rate += self.prob_handler.split_model().add_and_estimate_bit(0)
        rd.rate += self.prob_handler.block_model().add_and_estimate_bit(0)
        rd.distortion = energy(sub_block)
        return flags, rd

    def _estimate_integer(
        self,
        value: int,
        lower_bp: int,
        upper_bp: int,
        *,
        signed: bool,
    ) -> RD:
        lower_mask = (1 << lower_bp) - 1
        upper_mask = ~lower_mask
        rd = RD(
            rate=0,
            distortion=energy(np.abs(value) & lower_mask),
        )

        masked_value = np.abs(value) & upper_mask
        for i in range(lower_bp, upper_bp):
            bit = ((1 << i) & masked_value) != 0
            model = self.prob_handler.int_model(i)
            rd.rate += model.add_and_estimate_bit(bit)

        if signed and (masked_value != 0):
            model = self.prob_handler.signal_model()
            rd.rate += model.add_and_estimate_bit(value < 0)

        return rd

    @staticmethod
    def get_block_levels(block: np.ndarray) -> np.ndarray:
        """
        The levels of a (4, 5) block are organized as follows:

        0, 1, 2, 3, 4
        1, 1, 2, 3, 4
        2, 2, 2, 3, 4
        3, 3, 3, 3, 4
        """
        blocks_level = np.zeros_like(block, dtype=np.int32)
        for position, _ in np.ndenumerate(block):
            level = max(position)
            blocks_level[position] = level
        return blocks_level

    @staticmethod
    def find_bitplane_per_level(block: np.ndarray) -> Sequence[int]:
        """
        Find the maximum bitplane by level of the block.
        The code is a bit dumb, but it works and is fast enough.
        """

        tmp_block = block.copy()
        bitplane_sizes = []

        for i in range(max(block.shape)):
            slices = tuple(slice(0, min(i, dim)) for dim in range(block.ndim))
            tmp_block[*slices] = 0

            bp = MicoOptimizer.find_max_bitplane(tmp_block)
            bitplane_sizes.append(bp)

        return bitplane_sizes

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        """
        Find the minimum number of bits needed to encode all values of the block.
        It is the same thing as finding the bits needed to encode the maximum value.
        """
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()
