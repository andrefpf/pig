from functools import cache

import numpy as np
from bitarray import bitarray

from pig.entropy import CabacDecoder
from pig.utils.block_utils import bigger_possible_slice, split_shape_in_half

from .mico_probability_handler import MicoProbabilityHandler
from .utils import get_level, get_shape_levels, max_level


class MicoDecoder:
    """
    Multidimensional Image COdec - Decoder
    """

    def __init__(self):
        self.block: np.ndarray = np.array([], dtype=np.int32)
        self.block_levels: np.ndarray = np.array([], dtype=np.int32)
        self.level_bitplanes: np.ndarray = np.array([], dtype=np.int32)

        self.prob_handler = MicoProbabilityHandler()
        self.bitstream = bitarray()
        self.cabac = CabacDecoder()

    def decode(
        self,
        bitstream: bitarray,
        shape: tuple[int],
    ) -> np.ndarray:
        self.block = np.zeros(shape, dtype=np.int32)
        self.block_levels = get_shape_levels(shape)

        self.cabac.start(bitstream)
        self.decode_bitplane_sizes()
        self.apply_decoding(bigger_possible_slice(shape))
        self.cabac.end()
        return self.block

    def apply_decoding(self, block_position: tuple[slice]):
        sub_block = self.block[block_position]
        sub_levels = self.block_levels[block_position]
        upper_bitplanes = self.level_bitplanes[sub_levels]

        if np.all(upper_bitplanes <= self.lower_bitplane):
            return

        if np.all(upper_bitplanes <= 0):
            return

        flag = self.decode_flag(block_position)

        if flag == "S":  # Split
            for sub_pos in split_shape_in_half(block_position):
                self.apply_decoding(sub_pos)

        elif flag == "E":  # Empty
            assert sub_block.size != 1
            return

        elif flag == "F":  # Full
            for i, upper_bitplane in np.ndenumerate(upper_bitplanes):
                sub_block[i] = self.decode_int(
                    self.lower_bitplane,
                    upper_bitplane,
                    signed=True,
                )

        elif flag == "z":  # Zero
            assert sub_block.size == 1
            return

        elif flag == "v":  # Value
            assert sub_block.size == 1
            bitplane = self.get_bitplane(block_position)
            self.block[block_position] = self.decode_int(
                self.lower_bitplane,
                bitplane,
                signed=True,
            )

        else:
            raise ValueError(f'Invalid encoding flag "{flag}"')

    def decode_bitplane_sizes(self):
        self.lower_bitplane = self.decode_int(0, 5, signed=False)
        counter = self.lower_bitplane
        level_bitplanes = []

        for _ in range(max_level(self.block.shape)):
            while self.cabac.decode_bit(model=self.prob_handler.bitplanes_model()):
                counter += 1
            level_bitplanes.append(counter)

        level_bitplanes.reverse()
        self.level_bitplanes = np.array(level_bitplanes, dtype=np.int32)
        self.prob_handler.clear()

    def decode_int(
        self,
        lower_bitplane: int,
        upper_bitplane: int,
        *,
        signed: bool,
    ) -> int:
        value = 0

        for i in range(lower_bitplane, upper_bitplane):
            bit = self.cabac.decode_bit(model=self.prob_handler.int_model(i))
            value |= bit << i

        if signed and value != 0:
            signal = self.cabac.decode_bit(model=self.prob_handler.signal_model())
            if signal:
                value = -value

        return value

    def decode_flag(self, block_position: tuple[slice]):
        unitary = self.block[block_position].size == 1
        max_bp = self.get_bitplane(block_position)

        if unitary:
            unit_model = self.prob_handler.unit_model()
            unit_flag = self.cabac.decode_bit(model=unit_model)

            if unit_flag:
                return "v"
            else:
                return "z"

        else:
            significant_model = self.prob_handler.significant_model(max_bp)
            significant_flag = self.cabac.decode_bit(model=significant_model)

            if significant_flag:
                split_model = self.prob_handler.split_model(max_bp)
                split_flag = self.cabac.decode_bit(model=split_model)

                if split_flag:
                    return "S"
                else:
                    return "F"
            else:
                return "E"

    def get_bitplane(self, block_position: tuple[slice]):
        level = get_level(block_position)
        if level >= len(self.level_bitplanes):
            return self.level_bitplanes[-1]
        return self.level_bitplanes[level]
