from collections import deque

import numpy as np
from bitarray import bitarray

from pig.entropy import CabacEncoder, FrequentistPM
from pig.metrics import RD, energy
from pig.utils.block_utils import split_blocks_in_half

from .mule_optimizer import MuleOptimizer
from .mule_probability_handler import MuleProbabilityHandler

Flags = deque[str]


class MuleEncoder:
    def __init__(self):
        self.flags = ""
        self.estimated_rd = RD()

        self.lower_bitplane = 0
        self.upper_bitplane = 32
        self.lagrangian = 10_000

        self.optimizer = MuleOptimizer()
        self.prob_handler = MuleProbabilityHandler()
        self.bitstream = bitarray()
        self.cabac = CabacEncoder()

    def encode(
        self,
        block: np.ndarray,
        lagrangian: float = 10_000,
        *,
        upper_bitplane: int = None,
    ) -> bitarray:
        self.lagrangian = lagrangian
        self.optimizer.lagrangian = self.lagrangian

        if upper_bitplane is None:
            self.upper_bitplane = self.optimizer.find_max_bitplane(block)
        else:
            self.upper_bitplane = upper_bitplane

        self.lower_bitplane = self.optimizer.optimize_lower_bitplane(
            block,
            self.upper_bitplane,
        )

        self.flags, self.estimated_rd = self.optimizer.optimize_tree(
            block,
            self.lower_bitplane,
            self.upper_bitplane,
        )

        self.cabac.start(result=self.bitstream)
        self.encode_int(self.lower_bitplane, 0, 5, signed=False)
        self.apply_encoding(self.flags.copy(), block, self.upper_bitplane)

        return self.cabac.end(fill_to_byte=True)

    def apply_encoding(self, flags: Flags, block: np.ndarray, upper_bitplane: int):
        if upper_bitplane < self.lower_bitplane or upper_bitplane <= 0:
            return

        if block.size == 1:
            value = block.flatten()[0]
            self.encode_int(
                value,
                self.lower_bitplane,
                upper_bitplane,
                signed=True,
            )
            return

        flag = flags.popleft()
        model_0 = self.prob_handler.flag_model(upper_bitplane, 0)
        model_1 = self.prob_handler.flag_model(upper_bitplane, 1)

        if flag == "Z":
            # 1
            self.cabac.encode_bit(1, model=model_0)

        elif flag == "L":
            # 00
            self.cabac.encode_bit(0, model=model_0)
            self.cabac.encode_bit(0, model=model_1)
            self.apply_encoding(flags, block, upper_bitplane - 1)

        elif flag == "S":
            # 01
            self.cabac.encode_bit(0, model=model_0)
            self.cabac.encode_bit(1, model=model_1)
            for sub_block in split_blocks_in_half(block):
                self.apply_encoding(flags, sub_block, upper_bitplane)

        else:
            raise ValueError(f"Invalid flag {flag}")

    def encode_int(
        self,
        value: int,
        lower_bitplane: int,
        upper_bitplane: int,
        signed: bool,
    ):
        absolute = np.abs(value)
        for i in range(lower_bitplane, upper_bitplane):
            bit = ((1 << i) & absolute) != 0
            model = self.prob_handler.int_model(i)
            self.cabac.encode_bit(bit, model=model)

        mask = (1 << lower_bitplane) - 1
        if signed and (absolute & ~mask) != 0:
            model = self.prob_handler.signal_model()
            self.cabac.encode_bit(value < 0, model=model)
