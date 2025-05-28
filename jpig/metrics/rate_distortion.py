from dataclasses import dataclass


@dataclass
class RD:
    rate: float = 0
    distortion: float = 0

    def cost(self, lagrangian: float) -> float:
        return self.distortion + lagrangian * self.rate

    def __add__(self, other: "RD"):
        return RD(self.rate + other.rate, self.distortion + other.distortion)
