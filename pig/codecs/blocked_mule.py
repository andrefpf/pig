from math import ceil

import numpy as np
from bitarray import bitarray
from scipy.fft import dctn, idctn

from pig.entropy import MuleDecoder, MuleEncoder
from pig.metrics.rate_distortion import RD
from pig.utils.block_utils import split_blocks_equal_size


class BlockedMule:
    def encode(
        self,
        data: np.ndarray,
        lagrangian: float,
        block_size: int = 16,
        bitdepth: int = 8,
    ) -> bitarray:
        max_bitplane = data.ndim * (bitdepth - 1)

        bitstream = bitarray()
        block_encoded_sizes = []
        self.estimated_rd = RD()

        shifted = data.astype(np.int32) - (1 << (bitdepth - 1))
        for block in split_blocks_equal_size(shifted, block_size):
            mule_encoder = MuleEncoder()
            transformed_block: np.ndarray = dctn(block, norm="ortho")
            transformed_block = transformed_block.round().astype(int)
            block_bitstream = mule_encoder.encode(
                transformed_block,
                lagrangian,
                upper_bitplane=max_bitplane,
            )
            self.estimated_rd += mule_encoder.estimated_rd
            bitstream += block_bitstream
            block_encoded_sizes.append(len(block_bitstream) // 8)

        shape_bits = self._max_bits(data.shape)
        block_bits = self._max_bits(block_encoded_sizes)

        header = bitarray()
        self._add_bits(header, 8, data.ndim)
        self._add_bits(header, 8, shape_bits)
        for size in data.shape:
            self._add_bits(header, shape_bits, size)

        self._add_bits(header, 16, block_size)
        self._add_bits(header, 32, len(block_encoded_sizes))
        self._add_bits(header, 8, block_bits)
        for size in block_encoded_sizes:
            self._add_bits(header, block_bits, size)

        self._add_bits(header, 8, bitdepth)
        self._add_bits(header, 8, max_bitplane)
        return header + bitstream

    def decode(self, codestream: bitarray) -> np.ndarray:
        codestream = codestream.copy()
        ndim = self._consume_bits(codestream, 8)
        shape_bits = self._consume_bits(codestream, 8)
        shape = list()
        for _ in range(ndim):
            size = self._consume_bits(codestream, shape_bits)
            shape.append(size)

        block_size = self._consume_bits(codestream, 16)
        number_of_blocks = self._consume_bits(codestream, 32)
        block_bits = self._consume_bits(codestream, 8)

        block_encoded_sizes = []
        for _ in range(number_of_blocks):
            size = self._consume_bits(codestream, block_bits)
            block_encoded_sizes.append(size)

        bitdepth = self._consume_bits(codestream, 8)
        max_bitplane = self._consume_bits(codestream, 8)

        last_pos = 0
        bitstreams: list[bitarray] = list()
        for size in block_encoded_sizes:
            start = last_pos
            end = start + size * 8
            bitstreams.append(codestream[start:end])
            last_pos = end

        decoded = np.zeros(shape, dtype=int)
        for bitstream, block in zip(bitstreams, split_blocks_equal_size(decoded, block_size)):
            mule_decoder = MuleDecoder()
            transformed_block = mule_decoder.decode(
                bitstream,
                block.shape,
                upper_bitplane=max_bitplane,
            )
            decoded_block: np.ndarray = idctn(transformed_block, norm="ortho")
            decoded_block = decoded_block.round().astype(int)
            block[:] = decoded_block

        decoded += 1 << (bitdepth - 1)
        return decoded

    def _add_bits(self, codestram: bitarray, number_of_bits: int, value: int) -> None:
        bits = f"{value:0{number_of_bits}b}"
        codestram.extend(bits)

    def _consume_bits(self, codestream: bitarray, number_of_bits: int) -> int:
        section = codestream[:number_of_bits]
        left_fill = bitarray(section.padbits * "0")
        section = left_fill + section

        value = int.from_bytes(section.tobytes())
        codestream[:] = codestream[number_of_bits:]
        return value

    def _max_bits(self, sequence) -> int:
        return int(max(sequence)).bit_length()
