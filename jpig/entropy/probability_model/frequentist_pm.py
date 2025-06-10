from typing import Literal



from ._probability_model import ProbabilityModel

class FrequentistPM(ProbabilityModel):
    def probability(self, bit: bool | Literal[0, 1]) -> float:
        if self.frequency(0) <= 0:
            return 0

        if self.frequency(1) <= 0:
            return 0

        if bit:
            return self.frequency(1) / self.total_bits()
        else:
            return self.frequency(0) / self.total_bits()

    def push(self):
        self._stack.append((self._frequency_of_zeros, self._frequency_of_ones))

    def pop(self):
        self._frequency_of_zeros, self._frequency_of_ones = self._stack.pop()

    def __eq__(self, other) -> bool:
        return (
            self._frequency_of_zeros == other._frequency_of_zeros
            and self._frequency_of_ones == other._frequency_of_ones
        )
