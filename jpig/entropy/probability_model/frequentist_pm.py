from dataclasses import dataclass, field
from typing import Literal

import numpy as np


from ._probability_model import ProbabilityModel

@dataclass
class FrequentistPM(ProbabilityModel):
    frequency_of_zeros: int = 1
    frequency_of_ones: int = 1
    _stack: list[tuple[int, int]] = field(default_factory=list)

    def add_bit(self, bit: bool | Literal[0, 1]):
        if bit:
            self.frequency_of_ones += 1
        else:
            self.frequency_of_zeros += 1

    def push(self):
        self._stack.append((self.frequency_of_zeros, self.frequency_of_ones))

    def pop(self):
        self.frequency_of_zeros, self.frequency_of_ones = self._stack.pop()

    def clear(self):
        self.frequency_of_ones = 1
        self.frequency_of_zeros = 1
        self._stack.clear()

    def total_bits(self) -> int:
        return self.frequency_of_zeros + self.frequency_of_ones

    def frequency(self, bit: bool | Literal[0, 1]) -> int:
        return self.frequency_of_ones if bit else self.frequency_of_zeros

    def probability(self, bit: bool | Literal[0, 1]) -> float:
        if self.frequency(0) <= 0:
            return 0

        if self.frequency(1) <= 0:
            return 0

        if bit:
            return self.frequency_of_ones / self.total_bits()
        else:
            return self.frequency_of_zeros / self.total_bits()

    def entropy(self) -> float:
        prob_0 = self.probability(0)
        prob_1 = self.probability(1)

        if prob_0 == 0:
            return 0

        if prob_1 == 0:
            return 0

        return -prob_0 * np.log2(prob_0) - prob_1 * np.log2(prob_1)

    def estimated_rate(self) -> float:
        return self.total_bits() * self.entropy()
