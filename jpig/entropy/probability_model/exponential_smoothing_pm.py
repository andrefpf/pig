from typing import Literal

import numpy as np

from ._probability_model import ProbabilityModel


class ExponentialSmoothingPM(ProbabilityModel):
    def __init__(self, frequency_of_zeros=1, frequency_of_ones=1, smooth_factor=0.05):
        super().__init__(frequency_of_zeros, frequency_of_ones)

        self._probability_of_ones = self.frequency(0) / self.total_bits()
        self.smooth_factor = smooth_factor
        self.precision = 16
        self._min_probability = 1 / (1 << self.precision)
        self._max_probability = 1 - self._min_probability

    def add_bit(self, bit: bool | Literal[0, 1]) -> None:
        super().add_bit(bit)
        new_weight = self.smooth_factor * bit
        old_weight = (1 - self.smooth_factor) * self._probability_of_ones
        self._probability_of_ones = new_weight + old_weight

        self._probability_of_ones = np.clip(
            self._probability_of_ones,
            self._min_probability,
            self._max_probability,
        )

    def clear(self):
        super().clear()
        self._probability_of_ones = 0.5

    def probability(self, bit: bool | Literal[0, 1]) -> float:
        if bit:
            return self._probability_of_ones
        else:
            return 1 - self._probability_of_ones

    def push(self):
        data = (
            self._frequency_of_zeros,
            self._frequency_of_ones,
            self._probability_of_ones,
            self.smooth_factor,
        )
        self._stack.append(data)

    def apply(self):
        (
            self._frequency_of_zeros,
            self._frequency_of_ones,
            self._probability_of_ones,
            self.smooth_factor,
        ) = self._stack[-1]
