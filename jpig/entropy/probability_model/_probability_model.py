from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class ProbabilityModel(ABC):
    def __init__(self, frequency_of_zeros: int = 1, frequency_of_ones: int = 1):
        self._frequency_of_zeros = frequency_of_zeros
        self._frequency_of_ones = frequency_of_ones
        self._stack = list()

    def clear(self):
        self._frequency_of_ones = 1
        self._frequency_of_zeros = 1
        self._stack.clear()

    def add_bit(self, bit: bool | Literal[0, 1]) -> None:
        if bit:
            self._frequency_of_ones += 1
        else:
            self._frequency_of_zeros += 1

    def frequency(self, bit: bool | Literal[0, 1]) -> int:
        return self._frequency_of_ones if bit else self._frequency_of_zeros

    def total_bits(self) -> int:
        return self._frequency_of_zeros + self._frequency_of_ones

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

    @abstractmethod
    def probability(self, bit: bool | Literal[0, 1]) -> float: ...

    @abstractmethod
    def push(self): ...

    @abstractmethod
    def pop(self): ...
