from collections import deque
from typing import Sequence

import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacEncoder, FrequentistPM
from jpig.metrics import RD, energy
from jpig.utils.block_utils import split_blocks_in_half

from .mule_probability_handler import MuleProbabilityHandler

type Flags = deque[str]


class MuleOptimizer:
    def __init__(self, lagrangian: int = 10_000):
        self.lagrangian = lagrangian
        self.prob_handler = MuleProbabilityHandler()

    def optimize_lower_bitplane(self, block: np.ndarray, upper_bp: int) -> int:
        lower_bitplane = 0
        accumulated_rate = 0
        best_cost = float("inf")
        magnitudes = np.abs(block.flatten())

        for i in reversed(range(0, upper_bp)):
            bit_position = 1 << i
            mask = bit_position - 1
            model = self.prob_handler.int_model(i)
            non_zeroed = magnitudes > bit_position

            bits_to_encode = magnitudes[non_zeroed] & bit_position != 0
            for bit in bits_to_encode:
                accumulated_rate += model.add_and_estimate_bit(bit)

            sign_rate = np.sum(non_zeroed)
            total_rate = accumulated_rate + sign_rate
            rd = RD(total_rate, energy(magnitudes & mask))

            if rd.cost(self.lagrangian) < best_cost:
                best_cost = rd.cost(self.lagrangian)
                lower_bitplane = i

        return lower_bitplane

    def optimize_tree(
        self,
        block: np.ndarray,
        lower_bp: int,
        upper_bp: int,
    ) -> tuple[Flags, RD]:
        if upper_bp < lower_bp or upper_bp <= 0:
            rd = RD(
                rate=0,
                distortion=energy(block),
            )
            return (deque(), rd)

        if block.size == 1:
            value = block.flatten()[0]
            rd = self._estimate_integer(
                value,
                lower_bp,
                upper_bp,
                signed=True,
            )
            return (deque(), rd)

        self.prob_handler.push()
        should_lower_bitplane = self.is_bitplane_zero(block, upper_bp)
        if should_lower_bitplane:
            segmentation_flags, segmentation_rd = self._estimate_lower_bp(
                block,
                lower_bp,
                upper_bp,
            )
        else:
            segmentation_flags, segmentation_rd = self._estimate_split(
                block,
                lower_bp,
                upper_bp,
            )

        model = self.prob_handler.flag_model(upper_bp, 0)
        zero_rd = RD(
            rate=model.estimate_bit(1),
            distortion=energy(block),
        )

        if segmentation_rd.cost(self.lagrangian) < zero_rd.cost(self.lagrangian):
            return segmentation_flags, segmentation_rd
        else:
            self.prob_handler.pop()
            return self._estimate_zero(block, upper_bp)

    def _estimate_integer(
        self,
        value: int,
        lower_bp: int,
        upper_bp: int,
        signed: bool,
    ) -> RD:
        mask = (1 << lower_bp) - 1
        masked_value = np.abs(value) & mask

        rd = RD()
        rd.distortion += energy(masked_value)

        for i in range(lower_bp, upper_bp):
            bit = ((1 << i) & masked_value) != 0
            model = self.prob_handler.int_model(i)
            rd.rate += model.add_and_estimate_bit(bit)

        if signed and (masked_value != 0):
            model = self.prob_handler.signal_model()
            rd.rate += model.add_and_estimate_bit(value < 0)

        return rd

    def _estimate_lower_bp(
        self,
        block: np.ndarray,
        lower_bp: int,
        upper_bp: int,
    ) -> tuple[Flags, RD]:
        new_bp = self.find_max_bitplane(block)
        n_flags = upper_bp - new_bp

        flags = deque("L" * n_flags)
        rd = RD()

        model_0 = self.prob_handler.flag_model(upper_bp, 0)
        model_1 = self.prob_handler.flag_model(upper_bp, 1)

        for _ in range(n_flags):
            rd.rate += model_0.add_and_estimate_bit(0)
            rd.rate += model_1.add_and_estimate_bit(0)

        current_flags, current_rd = self.optimize_tree(
            block,
            lower_bp,
            new_bp,
        )

        flags += current_flags
        rd += current_rd

        return flags, rd

    def _estimate_split(
        self,
        block: np.ndarray,
        lower_bp: int,
        upper_bp: int,
    ) -> tuple[Flags, RD]:
        rd = RD()
        flags = deque("S")

        model_0 = self.prob_handler.flag_model(upper_bp, 0)
        model_1 = self.prob_handler.flag_model(upper_bp, 1)

        rd.rate += model_0.add_and_estimate_bit(0)
        rd.rate += model_1.add_and_estimate_bit(1)

        for sub_block in split_blocks_in_half(block):
            current_flags, current_rd = self.optimize_tree(
                sub_block,
                lower_bp,
                upper_bp,
            )
            rd += current_rd
            flags += current_flags

        return flags, rd

    def _estimate_zero(
        self,
        block: np.ndarray,
        upper_bp: int,
    ) -> tuple[Flags, RD]:
        flags = deque("Z")
        model = self.prob_handler.flag_model(upper_bp, 0)
        rd = RD(
            rate=model.add_and_estimate_bit(1),
            distortion=energy(block),
        )
        return flags, rd

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()

    @staticmethod
    def is_bitplane_zero(block, bitplane):
        return not np.any(np.abs(block) & 1 << (bitplane - 1))
