from dataclasses import dataclass
from functools import cache


@cache
def cost(
    rate: float,
    distortion: float,
    lagrangian: float,
) -> float:
    return distortion + lagrangian * rate


@dataclass
class RD:
    rate: float = 0
    distortion: float = 0

    def cost(self, lagrangian: float) -> float:
        return cost(self.rate, self.distortion, lagrangian)

    def __add__(self, other: "RD"):
        return RD(self.rate + other.rate, self.distortion + other.distortion)
