from abc import ABC, abstractmethod
from typing import Literal

class ProbabilityModel(ABC):
    @abstractmethod
    def add_bit(self, bit: bool | Literal[0, 1]) -> None:
        ...

    @abstractmethod
    def push(self):
        ...

    @abstractmethod
    def pop(self):
        ...

    @abstractmethod
    def clear(self):
        ...

    @abstractmethod
    def total_bits(self) -> int:
        ...

    @abstractmethod
    def frequency(self, bit: bool | Literal[0, 1]) -> int:
        ...

    @abstractmethod
    def probability(self, bit: bool | Literal[0, 1]) -> float:
        ...

    def entropy(self) -> float:
        ...

    def estimated_rate(self) -> float:
        return self.total_bits() * self.entropy()
