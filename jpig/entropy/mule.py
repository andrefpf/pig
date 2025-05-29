from copy import deepcopy
from typing import Sequence

import numpy as np
from jpig.metrics import RD, binary_entropy
from jpig.utils.block_utils import split_blocks_in_half

BITS_PER_FLAG = {
    "Z": 1,
    "L": 2,
    "S": 2,
}

MULE_REPRESENTATION = """Mule(
    rate_distortion=({rate} bits, {distortion:.2f} MSE),
    flags=[{flags}],
    signals=[{signals}],
    bitplanes=[
        {bitplanes}
    ],
)
"""


class Mule:
    def __init__(self):
        self.max_bitplane = -1
        self.rd = RD()
        self.flags: str = ""
        self.signals: list[bool] = list()
        self.encoded_bitplanes: list[list[bool]] = [[] for _ in range(32)]

    def encode(
        self,
        block: np.ndarray,
        lagrangian: float = 10_000,
        current_bitplane: int = 32,
    ):
        if block.size == 0:
            return self

        if block.size == 1:
            value = block.flatten()[0]
            return self._encode_value(value, current_bitplane)

        proportional_lagrangian = lagrangian / block.size

        zero_flag = Mule()._encode_zero_flag(block)
        zero_cost = zero_flag.rd.cost(proportional_lagrangian)
        if zero_flag.rd.distortion == 0:
            self += zero_flag
            return self

        if self.is_bitplane_zero(block, current_bitplane):
            lower_bp_flag = Mule()._encode_lower_bp_flag(
                block,
                lagrangian,
                current_bitplane,
            )
            lower_bp_cost = lower_bp_flag.rd.cost(proportional_lagrangian)
            if lower_bp_cost < zero_cost:
                self += lower_bp_flag
                return self

        split_flag = Mule()._encode_split_flag(block, lagrangian, current_bitplane)
        split_cost = split_flag.rd.cost(proportional_lagrangian)

        if split_cost < zero_cost:
            self += split_flag
        else:
            self += zero_flag

        return self

    def copy(self):
        return deepcopy(self)

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
        current_bitplane: int,
    ):
        """
        Add the "L" flag in the encoding.
        The added rate is just the cost of the flag, it does not introduces
        error, unless the recursive call introduces it.
        """

        self.flags = "L" + self.flags
        self.encode(block, lagrangian, current_bitplane - 1)
        self.rd.rate += BITS_PER_FLAG["L"]
        return self

    def _encode_split_flag(
        self,
        block: np.ndarray,
        lagrangian: float,
        current_bitplane: int,
    ):
        """
        Add the "S" flag in the encoding.
        The added rate is just the cost of the flag, it does not introduces
        error, unless the recursive call introduces it.
        """
        self.flags = "S" + self.flags
        for sub_block in split_blocks_in_half(block):
            self.encode(sub_block, lagrangian, current_bitplane)
        self.rd.rate = self._estimate_total_rate()
        return self

    def _encode_value(self, value: int, current_bitplane: int):
        self.signals.append(value < 0)
        for i in range(current_bitplane):
            self.encoded_bitplanes[i].append((1 << i) & value != 0)
        self.rd.rate = self._estimate_total_rate()
        return self

    def _estimate_total_rate(self):
        rate = binary_entropy(self.signals) * len(self.signals)
        for bitplane in self.encoded_bitplanes:
            rate += binary_entropy(bitplane) * len(bitplane)
        rate += sum(BITS_PER_FLAG[i] for i in self.flags)
        return rate

    def __str__(self) -> str:
        signals = "".join("-" if signal else "+" for signal in self.signals)
        bitplanes = []
        for i, bitplane in enumerate(self.encoded_bitplanes):
            if not bitplane:
                continue
            str_bitplane = f"({i}): " + "".join(["1" if i else "0" for i in bitplane])
            bitplanes.append(str_bitplane)
        str_bitplanes = "\n\t".join(bitplanes)

        return MULE_REPRESENTATION.format(
            # max_bitplane=self.max_bitplane,
            rate=np.ceil(self.rd.rate),
            distortion=self.rd.distortion,
            flags=self.flags,
            signals=signals,
            bitplanes=str_bitplanes,
        )

    def __add__(self, other: "Mule") -> "Mule":
        new = self.copy()
        new += other
        return new

    def __iadd__(self, other: "Mule") -> "Mule":
        self.max_bitplane = max(self.max_bitplane, other.max_bitplane)
        self.rd += other.rd
        self.flags += other.flags
        self.signals += other.signals
        for a, b in zip(self.encoded_bitplanes, other.encoded_bitplanes):
            a += b
        return self

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()

    @staticmethod
    def is_bitplane_zero(block, bitplane):
        return np.sum(np.abs(block) & 1 << (bitplane - 1)) == 0

    @staticmethod
    def _estimate_rate_of_sequence(sequence: Sequence[bool]):
        number_of_bits = len(sequence)
        number_of_one_bits = np.sum(sequence)
        number_of_zero_bits = number_of_bits - np.sum(sequence)

        if (number_of_bits * number_of_zero_bits * number_of_one_bits) == 0:
            return 0

        probability_of_one = number_of_one_bits / number_of_bits
        probability_of_zero = 1 - probability_of_one

        return np.log(probability_of_zero) * np.log(probability_of_one) * number_of_bits
