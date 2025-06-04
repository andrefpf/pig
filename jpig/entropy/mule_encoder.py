import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacEncoder, ProbabilityModel
from jpig.metrics import RD
from jpig.utils.block_utils import split_blocks_in_half

_Z = bitarray("0")
_L = bitarray("10")
_S = bitarray("11")


class MuleEncoder:
    def __init__(self):
        self.estimated_rd = RD()
        self.flags = ""

        self.upper_bitplane = 32
        self.lower_bitplane = 0
        self.lagrangian = 10_000

        self.flags_probability_model = ProbabilityModel()
        self.signals_probability_model = ProbabilityModel()
        self.bitplane_probability_models = [ProbabilityModel() for _ in range(32)]

        self.bitstream = bitarray()
        self.cabac = CabacEncoder()

    def encode(
        self,
        block: np.ndarray,
        lagrangian: float = 10_000,
        upper_bitplane: int = 32,
    ):
        self.lagrangian = lagrangian
        self.upper_bitplane = upper_bitplane

        self._clear_models()
        self.flags, self.estimated_rd = self._optimize_encoding_tree(block, self.upper_bitplane)
        self._clear_models()

        self.cabac.start(result=self.bitstream)
        self.apply_encoding(list(self.flags), block, self.upper_bitplane)
        return self.cabac.end()

    def apply_encoding(self, flags: list[str], block: np.ndarray, bitplane: int):
        if block.size == 1:
            value = block.flatten()[0]
            for i in range(self.lower_bitplane, bitplane):
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
            self.apply_encoding(flags, block, bitplane - 1)

        elif flag == "S":
            for bit in _S:
                self.cabac.encode_bit(bit, model=self.flags_probability_model)
            for sub_block in split_blocks_in_half(block):
                self.apply_encoding(flags, sub_block, bitplane)

        else:
            raise ValueError("Invalid encoding")

    def _optimize_encoding_tree(self, block: np.ndarray, bitplane: int) -> tuple[str, RD]:
        if block.size == 1:
            return "", self._estimate_value_encoding(block, bitplane)

        proportional_lagrangian = self.lagrangian / block.size

        zero_flags, zero_rd = self._estimate_zero_flag(block)
        zero_cost = zero_rd.cost(proportional_lagrangian)
        if zero_rd.distortion == 0:
            return zero_flags, zero_rd

        if self.is_bitplane_zero(block, bitplane):
            self._push_models()
            lower_bp_flags, lower_bp_rd = self._estimate_lower_bp_flag(block, bitplane)
            lower_bp_cost = lower_bp_rd.cost(proportional_lagrangian)
            if lower_bp_cost < zero_cost:
                return lower_bp_flags, lower_bp_rd
            self._pop_models()
            return zero_flags, zero_rd

        self._push_models()
        split_flags, split_rd = self._estimate_split_flag(block, bitplane)
        split_cost = split_rd.cost(proportional_lagrangian)
        if split_cost < zero_cost:
            return split_flags, split_rd
        self._pop_models()
        return zero_flags, zero_rd

    def _estimate_zero_flag(self, block: np.ndarray) -> tuple[str, RD]:
        rate = len(_Z)
        distortion = np.sum(block**2)
        return "Z", RD(rate, distortion)

    def _estimate_lower_bp_flag(self, block: np.ndarray, bitplane: int) -> tuple[str, RD]:
        new_bitplane = self.find_max_bitplane(block)
        flags, rd = self._optimize_encoding_tree(block, new_bitplane)
        number_of_flags = bitplane - new_bitplane
        rd.rate += len(_L) * number_of_flags
        return ("L" * number_of_flags + flags), rd

    def _estimate_split_flag(self, block: np.ndarray, bitplane: int) -> tuple[str, RD]:
        rd = RD()
        rd.rate += len(_S)
        flags = "S"
        for sub_block in split_blocks_in_half(block):
            current_flags, current_rd = self._optimize_encoding_tree(sub_block, bitplane)
            rd += current_rd
            flags += current_flags
        return flags, rd

    def _estimate_value_encoding(self, block: np.ndarray, bitplane: int) -> RD:
        value = block.flatten()[0]
        for i in range(self.lower_bitplane, bitplane):
            model = self.bitplane_probability_models[i]
            model.add_bit((1 << i) & value)
        rate = self._estimate_current_rate()
        return RD(rate, 0)

    def _estimate_current_rate(self) -> float:
        total_size = 0
        for model in self.bitplane_probability_models:
            total_size += model.total_bits() * model.entropy()
        return total_size

    def _push_models(self):
        for model in self.bitplane_probability_models:
            model.push()

    def _pop_models(self):
        for model in self.bitplane_probability_models:
            model.pop()

    def _clear_models(self):
        for model in self.bitplane_probability_models:
            model.clear()

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()

    @staticmethod
    def is_bitplane_zero(block, bitplane):
        return np.sum(np.abs(block) & 1 << (bitplane - 1)) == 0
