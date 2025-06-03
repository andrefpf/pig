from dataclasses import dataclass
import numpy as np
from typing import Literal


@dataclass
class ProbabilityModel:
    frequency_of_zeros: int = 1
    frequency_of_ones: int = 1

    def add_bit(self, bit: bool | Literal[0, 1]):
        if bit:
            self.frequency_of_ones += 1
        else:
            self.frequency_of_zeros += 1

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
        prob_0 = self.probability_of_zeros()
        prob_1 = self.probability_of_ones()

        if prob_0 == 0:
            return 0

        if prob_1 == 0:
            return 1

        return -prob_0 * np.log2(prob_0) - prob_1 * np.log2(prob_1)
