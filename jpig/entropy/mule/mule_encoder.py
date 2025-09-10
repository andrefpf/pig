from typing import Sequence

import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacEncoder, FrequentistPM
from jpig.metrics import RD, energy
from jpig.utils.block_utils import split_blocks_in_half

from .mule_probability_handler import MuleProbabilityHandler


class MuleEncoder:
    def __init__(self):
        self.estimated_rd = RD()
        self.flags = ""

        self.lower_bitplane = 0
        self.upper_bitplane = 32
        self.lagrangian = 10_000

        self.prob_handler = MuleProbabilityHandler()
        self.bitstream = bitarray()
        self.cabac = CabacEncoder()

    def encode(
        self,
        block: np.ndarray,
        lagrangian: float = 10_000,
        *,
        lower_bitplane: int = None,
        upper_bitplane: int = None,
    ) -> bitarray:
        self.lagrangian = lagrangian
        self.upper_bitplane = upper_bitplane
        self.lower_bitplane = lower_bitplane

        if self.upper_bitplane is None:
            self.upper_bitplane = self.find_max_bitplane(block)

        if self.lower_bitplane is None:
            self.lower_bitplane = self._find_optimal_lower_bitplane(block)

        self.prob_handler.clear()
        self.flags, self.estimated_rd = self._recursive_optimize_encoding_tree(
            block,
            self.upper_bitplane,
        )
        self.prob_handler.clear()

        self.cabac.start(result=self.bitstream)
        self.encode_int(self.lower_bitplane, 0, 5, signed=False)
        self.apply_encoding(list(self.flags), block, self.upper_bitplane)
        return self.cabac.end(fill_to_byte=True)

    def apply_encoding(self, flags: Sequence[str], block: np.ndarray, upper_bitplane: int):
        if upper_bitplane < self.lower_bitplane or upper_bitplane <= 0:
            return

        if block.size == 1:
            value = block.flatten()[0]
            self.encode_int(value, self.lower_bitplane, upper_bitplane)
            return

        flag = flags.pop(0)

        if flag == "Z":
            # 1
            self.cabac.encode_bit(1, model=self.prob_handler.flag_model(upper_bitplane, 0))

        elif flag == "L":
            # 00
            self.cabac.encode_bit(0, model=self.prob_handler.flag_model(upper_bitplane, 0))
            self.cabac.encode_bit(0, model=self.prob_handler.flag_model(upper_bitplane, 1))
            self.apply_encoding(flags, block, upper_bitplane - 1)

        elif flag == "S":
            # 01
            self.cabac.encode_bit(0, model=self.prob_handler.flag_model(upper_bitplane, 0))
            self.cabac.encode_bit(1, model=self.prob_handler.flag_model(upper_bitplane, 1))
            for sub_block in split_blocks_in_half(block):
                self.apply_encoding(flags, sub_block, upper_bitplane)

        else:
            raise ValueError("Invalid encoding")

    def encode_int(
        self,
        value: int,
        lower_bitplane: int,
        upper_bitplane: int,
        signed: bool = True,
    ):
        absolute = np.abs(value)
        for i in range(lower_bitplane, upper_bitplane):
            bit = ((1 << i) & absolute) != 0
            self.cabac.encode_bit(bit, model=self.prob_handler.int_model(i))

        mask = (1 << lower_bitplane) - 1
        if signed and (absolute & ~mask) != 0:
            self.cabac.encode_bit(value < 0, model=self.prob_handler.signal_model())

    def _find_optimal_lower_bitplane(self, block: np.ndarray) -> int:
        lower_bitplane = 0
        accumulated_rate = 0
        best_cost = float("inf")
        magnitudes = np.abs(block.flatten())

        for i in reversed(range(0, self.upper_bitplane)):
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

    def _recursive_optimize_encoding_tree(
        self,
        block: np.ndarray,
        upper_bitplane: int,
    ) -> tuple[str, RD]:
        if upper_bitplane < self.lower_bitplane or upper_bitplane <= 0:
            rd = RD(
                rate=0,
                distortion=energy(block),
            )
            return "", rd

        if block.size == 1:
            rd = self._estimate_integer(block, self.lower_bitplane, upper_bitplane)
            return "", rd

        self.prob_handler.push()
        should_lower_bitplane = self.is_bitplane_zero(block, upper_bitplane)
        if should_lower_bitplane:
            segmentation_flags, segmentation_rd = self._estimate_lower_bp_flag(
                block,
                upper_bitplane,
            )
        else:
            segmentation_flags, segmentation_rd = self._estimate_split_flag(
                block,
                upper_bitplane,
            )

        zero_rd = RD(
            rate=self.prob_handler.flag_model(upper_bitplane, 0).estimate_bit(1),
            distortion=energy(block),
        )

        if segmentation_rd.cost(self.lagrangian) < zero_rd.cost(self.lagrangian):
            return segmentation_flags, segmentation_rd
        else:
            self.prob_handler.pop()
            return self._estimate_zero_flag(block, upper_bitplane)

    def _estimate_zero_flag(self, block: np.ndarray, upper_bitplane: int) -> tuple[str, RD]:
        rd = RD()
        rd.distortion = energy(block)
        rd.rate += self.prob_handler.flag_model(upper_bitplane, 0).add_and_estimate_bit(1)
        return "Z", rd

    def _estimate_lower_bp_flag(self, block: np.ndarray, upper_bitplane: int) -> tuple[str, RD]:
        new_bitplane = self.find_max_bitplane(block)
        number_of_flags = upper_bitplane - new_bitplane

        flags = "L" * number_of_flags
        rd = RD()

        for _ in range(number_of_flags):
            rd.rate += self.prob_handler.flag_model(upper_bitplane, 0).add_and_estimate_bit(0)
            rd.rate += self.prob_handler.flag_model(upper_bitplane, 1).add_and_estimate_bit(0)

        current_flags, current_rd = self._recursive_optimize_encoding_tree(
            block,
            new_bitplane,
        )

        flags += current_flags
        rd += current_rd

        return flags, rd

    def _estimate_split_flag(
        self,
        block: np.ndarray,
        upper_bitplane: int,
    ) -> tuple[str, RD]:
        rd = RD()
        flags = "S"

        rd.rate += self.prob_handler.flag_model(upper_bitplane, 0).add_and_estimate_bit(0)
        rd.rate += self.prob_handler.flag_model(upper_bitplane, 1).add_and_estimate_bit(1)

        for sub_block in split_blocks_in_half(block):
            current_flags, current_rd = self._recursive_optimize_encoding_tree(
                sub_block,
                upper_bitplane,
            )
            rd += current_rd
            flags += current_flags

        return flags, rd

    def _estimate_integer(
        self,
        block: np.ndarray,
        lower_bitplane: int,
        upper_bitplane: int,
    ) -> RD:
        value = block.flatten()[0]
        mask = (1 << lower_bitplane) - 1
        masked_value = np.abs(value) & mask

        rd = RD()
        rd.distortion += energy(masked_value)

        for i in range(lower_bitplane, upper_bitplane):
            bit = (1 << i) & masked_value != 0
            model = self.prob_handler.int_model(i)
            rd.rate += model.add_and_estimate_bit(bit)

        if masked_value != 0:
            rd.rate += self.prob_handler.signal_model().add_and_estimate_bit(value < 0)

        return rd

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()

    @staticmethod
    def is_bitplane_zero(block, bitplane):
        return not np.any(np.abs(block) & 1 << (bitplane - 1))
