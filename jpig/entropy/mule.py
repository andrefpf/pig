import numpy as np
from jpig.metrics.rate_distortion import RD
from jpig.utils.block_utils import split_blocks_in_half

BITS_PER_FLAG = {
    "Z": 1,
    "L": 2,
    "S": 2,
}


class Mule:
    def __init__(self):
        self.max_bitplane = 0
        self.encoded_bitplanes: list[list[bool]] = [[] for _ in range(32)]
        self.signals: list[bool] = list()
        self.flags: str = ""
        self.rd = RD()

    def encode(
        self,
        block: np.ndarray,
        lagrangian: float,
        bitplane: int = -1,
    ):
        if self.max_bitplane < 0:
            # It just enters here on the first call
            bitplane = self.find_max_bitplane(block)
            self.max_bitplane = bitplane

        if block.size == 0:
            return self

        if block.size == 1:
            value = block.flatten()[0]
            return self._encode_value(value, bitplane)

        zero_flag = Mule()._encode_zero_flag(block)
        if zero_flag.rd.distortion == 0:
            return zero_flag

        lower_bp_flag = Mule()._encode_lower_bp_flag(block, lagrangian, bitplane)
        split_flag = Mule()._encode_split_flag(block, lagrangian, bitplane)

        proportional_lagrangian = lagrangian / block.size
        zero_cost = zero_flag.rd.cost(proportional_lagrangian)
        lower_bp_cost = lower_bp_flag.rd.cost(proportional_lagrangian)
        split_cost = split_flag.rd.cost(proportional_lagrangian)
        min_cost = min(zero_cost, lower_bp_cost, split_cost)

        if min_cost == zero_cost:
            return zero_cost
        elif min_cost == lower_bp_cost:
            return lower_bp_cost
        else:
            return split_cost

    def _encode_zero_flag(self, block: np.ndarray):
        """
        Add the "Z" flag in the encoding.
        The added rate is just the cost of the flag, and the distortion is the
        MSE of the block when compared to the zeroed output, i.e. the block squared.
        """
        self.flags = "Z" + self.flags
        self.rd.rate += BITS_PER_FLAG["Z"]
        self.rd.distortion += np.sum(block**2)
        return self

    def _encode_lower_bp_flag(
        self,
        block: np.ndarray,
        lagrangian: float,
        bitplane: int,
    ):
        """
        Add the "Z" flag in the encoding.
        The added rate is just the cost of the flag, it does not introduces
        error, unless the recursive call introduces it or the current bitplane
        was not zeroed.
        """

        self.flags = "L" + self.flags
        self.rd.rate += BITS_PER_FLAG["L"]
        self.rd.distortion += np.sum((block & (1 << bitplane)) ** 2)
        self.encode(block, lagrangian, bitplane - 1)
        return self

    def _encode_split_flag(
        self,
        block: np.ndarray,
        lagrangian: float,
        bitplane: int,
    ):
        """
        Add the "S" flag in the encoding.
        The added rate is just the cost of the flag, it does not introduces
        error, unless the recursive call introduces it.
        """
        self.flags = "S" + self.flags
        self.rd.rate += BITS_PER_FLAG["S"]
        for sub_block in split_blocks_in_half(block):
            self.encode(sub_block, lagrangian, bitplane)
        return self

    def _encode_value(self, value: int, bitplane: int):
        for i in range(bitplane):
            self.signals.append(value > 0)
            self.encoded_bitplanes[i].append((1 << bitplane) & value != 0)
        self.rd.rate += self._estimate_rate()
        return self

    def _estimate_rate(self):
        rate = 0
        rate += -np.log2(np.sum(self.signals) / len(self.signals))
        for bitplane_values in self.encoded_bitplanes:
            rate += -np.log2(np.sum(bitplane_values) / len(bitplane_values))
        return rate

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()

    @staticmethod
    def is_bitplane_zero(block, bitplane):
        return np.sum(np.abs(block) & 1 << (bitplane - 1)) == 0
