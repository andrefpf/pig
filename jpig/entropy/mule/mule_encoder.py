from typing import Sequence

import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacEncoder, FrequentistPM
from jpig.metrics import RD
from jpig.utils.block_utils import split_blocks_in_half

_Z = bitarray("0")
_L = bitarray("10")
_S = bitarray("11")


class MuleEncoder:
    def __init__(self):
        self.estimated_rd = RD()
        self.flags = ""

        self.lower_bitplane = 0
        self.upper_bitplane = 32
        self.lagrangian = 10_000

        self.flags_probability_model = FrequentistPM()
        self.signals_probability_model = FrequentistPM()
        self.bitplane_probability_models = [FrequentistPM() for _ in range(32)]

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
        self.lagrangian = lagrangian
        self.lower_bitplane = lower_bitplane
        self.upper_bitplane = upper_bitplane

        self._clear_models()
        self.flags, estimated_distortion = self._recursive_optimize_encoding_tree(
            block,
            lower_bitplane,
            upper_bitplane,
        )
        self.estimated_rd = RD(self._estimate_current_rate(), estimated_distortion)
        self._clear_models()

        self.cabac.start(result=self.bitstream)
        self.apply_encoding(list(self.flags), block, self.upper_bitplane)
        return self.cabac.end(fill_to_byte=True)

    def apply_encoding(
        self, flags: Sequence[str], block: np.ndarray, upper_bitplane: int
    ):
        if block.size == 1:
            value = block.flatten()[0]
            for i in range(self.lower_bitplane, upper_bitplane):
                bit = (1 << i) & np.abs(value) != 0
                self.cabac.encode_bit(bit, model=self.bitplane_probability_models[i])
            self.cabac.encode_bit(value < 0, model=self.signals_probability_model)
            return

        flag = flags.pop(0)

        if flag == "Z":
            for bit in _Z:
                self.cabac.encode_bit(bit, model=self.flags_probability_model)

        elif flag == "L":
            for bit in _L:
                self.cabac.encode_bit(bit, model=self.flags_probability_model)
            self.apply_encoding(flags, block, upper_bitplane - 1)

        elif flag == "S":
            for bit in _S:
                self.cabac.encode_bit(bit, model=self.flags_probability_model)
            for sub_block in split_blocks_in_half(block):
                self.apply_encoding(flags, sub_block, upper_bitplane)

        else:
            raise ValueError("Invalid encoding")

    def _recursive_optimize_encoding_tree(
        self,
        block: np.ndarray,
        lower_bitplane: int,
        upper_bitplane: int,
    ) -> tuple[str, float]:
        if block.size == 1:
            self._update_models_with_value(block, lower_bitplane, upper_bitplane)
            return "", 0

        lagrangian = self.lagrangian

        zero_flags, zero_distortion = self._estimate_zero_flag(block)
        zero_rd = RD(self._estimate_current_rate(), zero_distortion)
        if zero_rd.distortion == 0:
            return zero_flags, zero_distortion

        if self.is_bitplane_zero(block, upper_bitplane):
            self._push_models()
            lower_bp_flags, lower_bp_distortion = self._estimate_lower_bp_flag(
                block,
                lower_bitplane,
                upper_bitplane,
            )
            lower_bp_rd = RD(self._estimate_current_rate(), lower_bp_distortion)
            if lower_bp_rd.cost(lagrangian) < zero_rd.cost(lagrangian):
                return lower_bp_flags, lower_bp_distortion
            self._pop_models()
            return zero_flags, zero_distortion

        self._push_models()
        split_flags, split_distortion = self._estimate_split_flag(
            block,
            lower_bitplane,
            upper_bitplane,
        )
        split_rd = RD(self._estimate_current_rate(), split_distortion)
        if split_rd.cost(lagrangian) < zero_rd.cost(lagrangian):
            return split_flags, split_distortion
        self._pop_models()
        return zero_flags, zero_distortion

    def _estimate_zero_flag(self, block: np.ndarray) -> tuple[str, float]:
        distortion = np.sum(block.astype(np.int64) ** 2)
        return "Z", distortion

    def _estimate_lower_bp_flag(
        self, block: np.ndarray, lower_bitplane: int, upper_bitplane: int
    ) -> tuple[str, float]:
        new_bitplane = self.find_max_bitplane(block)
        number_of_flags = upper_bitplane - new_bitplane
        flags, distortion = self._recursive_optimize_encoding_tree(
            block,
            lower_bitplane,
            new_bitplane,
        )
        new_flags = "L" * number_of_flags + flags
        return new_flags, distortion

    def _estimate_split_flag(
        self, block: np.ndarray, lower_bitplane: int, upper_bitplane: int
    ) -> tuple[str, float]:
        distortion = 0
        flags = "S"
        for sub_block in split_blocks_in_half(block):
            current_flags, current_distortion = self._recursive_optimize_encoding_tree(
                sub_block,
                lower_bitplane,
                upper_bitplane,
            )
            distortion += current_distortion
            flags += current_flags
        return flags, distortion

    def _update_models_with_value(
        self,
        block: np.ndarray,
        lower_bitplane: int,
        upper_bitplane: int,
    ):
        value = block.flatten()[0]
        for i in range(lower_bitplane, upper_bitplane):
            model = self.bitplane_probability_models[i]
            model.add_bit((1 << i) & np.abs(value) != 0)

    def _estimate_current_rate(self) -> float:
        all_models = [
            self.flags_probability_model,
            self.signals_probability_model,
            *self.bitplane_probability_models,
        ]

        total_size = 0
        for model in all_models:
            total_size += model.total_estimated_rate()
        return total_size

    def _push_models(self):
        for model in self.probability_models():
            model.push()

    def _pop_models(self):
        for model in self.probability_models():
            model.pop()

    def _clear_models(self):
        for model in self.probability_models():
            model.clear()

    def probability_models(self):
        return [
            self.flags_probability_model,
            self.signals_probability_model,
            *self.bitplane_probability_models,
        ]

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()

    @staticmethod
    def is_bitplane_zero(block, bitplane):
        return np.sum(np.abs(block) & 1 << (bitplane - 1)) == 0
