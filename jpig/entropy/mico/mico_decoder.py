from functools import cache

import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacDecoder, FrequentistPM
from jpig.utils.block_utils import bigger_possible_slice, split_shape_in_half

from .mico_probability_handler import MicoProbabilityHandler


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
        self.block_levels = MicoDecoder.get_shape_levels(shape)

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
            bitplane = self._get_bitplane(block_position)
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

        for _ in range(max(self.block.shape)):
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
        max_bp = self._get_bitplane(block_position)

        if unitary:
            unit_flag = self.cabac.decode_bit(model=self.prob_handler.unit_model())
            if unit_flag:
                return "v"
            else:
                return "z"

        else:
            split_flag = self.cabac.decode_bit(model=self.prob_handler.split_model(max_bp))
            if split_flag:
                return "S"
            else:
                block_flag = self.cabac.decode_bit(model=self.prob_handler.block_model(max_bp))
                if block_flag:
                    return "F"
                else:
                    return "E"

    def _get_bitplane(self, block_position: tuple[slice]):
        level = max(s.start for s in block_position)
        return self.level_bitplanes[level]

    @staticmethod
    def get_shape_levels(shape: tuple[int, ...]) -> np.ndarray:
        """
        The levels of a (4, 5) block are organized as follows:

        0, 1, 2, 3, 4
        1, 1, 2, 3, 4
        2, 2, 2, 3, 4
        3, 3, 3, 3, 4
        """

        # I know it is dumb to cache and return a copy,
        # but python iterations are slow, numpy is fast.
        return _cached_shape_levels(shape).copy()


@cache
def _cached_shape_levels(shape: tuple[int, ...]) -> np.ndarray:
    blocks_level = np.zeros(shape, dtype=np.int32)
    for position in np.ndindex(*shape):
        level = max(position)
        blocks_level[position] = level
    return blocks_level
