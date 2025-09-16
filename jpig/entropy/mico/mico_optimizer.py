from collections import deque
from functools import cache
from typing import Sequence

import numpy as np

from jpig.metrics import RD, energy
from jpig.utils.block_utils import bigger_possible_slice, split_shape_in_half

from .mico_probability_handler import MicoProbabilityHandler
from .utils import get_block_levels, get_level, max_level

type Flags = deque[str]


class MicoOptimizer:
    def __init__(self, block: np.ndarray, lagrangian: int = 10_000):
        self.block = block
        self.lagrangian = lagrangian
        self.prob_handler = MicoProbabilityHandler()

        self.block_levels = get_block_levels(self.block)
        self.level_bitplanes = MicoOptimizer.find_bitplane_per_level(self.block)
        self.lower_bitplane = self.optimize_lower_bitplane()

    def optimize_lower_bitplane(self) -> int:
        lower_bitplane = 0
        accumulated_rate = 0
        best_cost = float("inf")

        magnitudes = np.abs(self.block).flatten()
        upper_bp = max(self.level_bitplanes)

        for i in reversed(range(0, upper_bp)):
            bit_position = 1 << i
            lower_mask = bit_position - 1
            non_zeroed = magnitudes >= bit_position

            model = self.prob_handler.int_model(i)
            bits_to_encode = magnitudes[non_zeroed] & bit_position != 0
            for bit in bits_to_encode:
                accumulated_rate += model.add_and_estimate_bit(bit)

            sign_rate = np.sum(non_zeroed, dtype=np.float64)
            total_rate = accumulated_rate + sign_rate
            rd = RD(total_rate, energy(magnitudes & lower_mask))

            if rd.cost(self.lagrangian) < best_cost:
                best_cost = rd.cost(self.lagrangian)
                lower_bitplane = i

        self.prob_handler.clear()
        return lower_bitplane

    def optimize_tree(self, block_position: tuple[slice, ...] | None = None) -> tuple[Flags, RD]:
        if block_position is None:
            block_position = bigger_possible_slice(self.block.shape)

        sub_block = self.block[block_position]
        sub_levels = self.block_levels[block_position]
        upper_bitplanes = self.level_bitplanes[sub_levels]

        if np.all(upper_bitplanes <= self.lower_bitplane) or np.all(upper_bitplanes <= 0):
            rd = RD(
                rate=0,
                distortion=energy(sub_block),
            )
            return (deque(), rd)

        if sub_block.size == 1:
            return self._estimate_unit_block(block_position)

        if np.all(sub_block == 0):
            return self._estimate_empty(block_position)

        self.prob_handler.push()
        _, empty_rd = self._estimate_empty(block_position)
        self.prob_handler.pop()

        self.prob_handler.push()
        _, full_rd = self._estimate_full(block_position)
        self.prob_handler.pop()

        self.prob_handler.push()
        split_flags, split_rd = self._estimate_split(block_position)

        split_cost = split_rd.cost(self.lagrangian)
        empty_cost = empty_rd.cost(self.lagrangian)
        full_cost = full_rd.cost(self.lagrangian)

        if split_cost < empty_cost and split_cost < full_cost:
            return split_flags, split_rd

        elif empty_cost < full_cost:
            self.prob_handler.pop()
            return self._estimate_empty(block_position)

        else:
            return self._estimate_full(block_position)

    def _estimate_unit_block(self, block_position: tuple[slice, ...]) -> tuple[Flags, RD]:
        sub_block = self.block[block_position]
        sub_levels = self.block_levels[block_position]

        value = sub_block.flatten()[0]
        level = sub_levels.flatten()[0]
        lower_bp = self.lower_bitplane
        upper_bp = self.level_bitplanes[level]

        lower_mask = (1 << lower_bp) - 1
        upper_mask = ~lower_mask
        quantized_value = np.abs(value) & upper_mask
        model = self.prob_handler.unit_model()

        flags = deque("v") if (quantized_value != 0) else deque("z")
        rd = RD()
        rd.rate += model.add_and_estimate_bit(quantized_value != 0)
        rd += self._estimate_integer(
            value,
            lower_bp,
            upper_bp,
            signed=True,
        )

        return (flags, rd)

    def _estimate_empty(self, block_position: tuple[slice, ...]) -> tuple[Flags, RD]:
        flags = deque("E")
        sub_block = self.block[block_position]
        max_bp = self.get_bitplane(block_position)

        rd = RD()
        rd.rate += self.prob_handler.significant_model(max_bp).add_and_estimate_bit(0)
        rd.distortion = energy(sub_block)
        return flags, rd

    def _estimate_full(
        self,
        block_position: tuple[slice, ...],
    ) -> tuple[Flags, RD]:
        flags = deque("F")
        sub_block = self.block[block_position]
        sub_levels = self.block_levels[block_position]
        lower_bp = self.lower_bitplane
        max_bp = self.get_bitplane(block_position)

        rd = RD()
        rd.rate += self.prob_handler.significant_model(max_bp).add_and_estimate_bit(1)
        rd.rate += self.prob_handler.split_model(max_bp).add_and_estimate_bit(0)

        for level, value in zip(sub_levels.flatten(), sub_block.flatten()):
            upper_bp = self.level_bitplanes[level]
            rd += self._estimate_integer(value, lower_bp, upper_bp, signed=True)

        return flags, rd

    def _estimate_split(self, block_position: tuple[slice, ...]) -> tuple[Flags, RD]:
        max_bp = self.get_bitplane(block_position)

        rd = RD()
        rd.rate += self.prob_handler.significant_model(max_bp).add_and_estimate_bit(1)
        rd.rate += self.prob_handler.split_model(max_bp).add_and_estimate_bit(1)

        flags = deque("S")
        for sub_pos in split_shape_in_half(block_position):
            current_flags, current_rd = self.optimize_tree(sub_pos)
            flags += current_flags
            rd += current_rd

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

        rd = RD()
        rd.distortion = energy(np.abs(value) & lower_mask)

        quantized_value = np.abs(value) & upper_mask
        for i in range(lower_bp, upper_bp):
            bit = ((1 << i) & quantized_value) != 0
            model = self.prob_handler.int_model(i)
            rd.rate += model.add_and_estimate_bit(bit)

        if signed and (quantized_value != 0):
            model = self.prob_handler.signal_model()
            rd.rate += model.add_and_estimate_bit(value < 0)

        return rd

    def get_bitplane(self, block_position: tuple[slice]):
        level = get_level(block_position)
        if level >= len(self.level_bitplanes):
            return self.level_bitplanes[-1]
        return self.level_bitplanes[level]

    @staticmethod
    def find_bitplane_per_level(block: np.ndarray) -> np.ndarray:
        """
        Find the maximum bitplane by level of the block.
        The code is a bit dumb, but it works and is fast enough.
        """

        total_levels = max_level(block.shape)
        bitplane_sizes = np.array([0 for _ in range(total_levels)], dtype=np.int32)

        for position, value in np.ndenumerate(block):
            level = min(get_level(position), total_levels - 1)
            max_bp = int(value).bit_length()
            bitplane_sizes[level] = max(bitplane_sizes[level], max_bp)

        # Each level is assumed to have equal or smaller bitplane size
        bitplane_sizes = np.maximum.accumulate(bitplane_sizes[::-1])[::-1]
        return bitplane_sizes

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        """
        Find the minimum number of bits needed to encode all values of the block.
        It is the same thing as finding the bits needed to encode the maximum value.
        """
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()
