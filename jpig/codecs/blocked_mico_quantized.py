from math import ceil

import numpy as np
from bitarray import bitarray
from scipy.fft import dctn, idctn

from jpig.entropy import MicoDecoder, MicoEncoder
from jpig.utils.block_utils import split_blocks_equal_size


class BlockedMicoQuantized:
    def encode(self, data: np.ndarray, lagrangian: float, quality: int, block_size: int = 8) -> bitarray:
        bitstream = bitarray()
        block_encoded_sizes = []

        quantization_matrix = self._get_quantization_matrix(block_size, data.ndim, quality=quality)

        for block in split_blocks_equal_size(data, block_size):
            transformed_block: np.ndarray = dctn(block, norm="ortho") / quantization_matrix
            transformed_block = transformed_block.round().astype(int)

            mico_encoder = MicoEncoder()
            block_bitstream = mico_encoder.encode(
                transformed_block,
                lagrangian,
            )
            bitstream += block_bitstream
            block_encoded_sizes.append(len(block_bitstream) // 8)

        header = bitarray()
        header.extend(f"{data.ndim:08b}")
        for size in data.shape:
            header.extend(f"{size:032b}")
        header.extend(f"{quality:08b}")

        header.extend(f"{block_size:016b}")
        header.extend(f"{len(block_encoded_sizes):032b}")
        for size in block_encoded_sizes:
            header.extend(f"{size:032b}")

        return header + bitstream

    def decode(self, codestream: bitarray) -> np.ndarray:
        codestream = codestream.copy()
        ndim = self._consume_bytes(codestream, 8)
        shape = list()
        for _ in range(ndim):
            size = self._consume_bytes(codestream, 32)
            shape.append(size)

        quality = self._consume_bytes(codestream, 8)

        block_size = self._consume_bytes(codestream, 16)
        number_of_blocks = self._consume_bytes(codestream, 32)

        block_encoded_sizes = []
        for _ in range(number_of_blocks):
            size = self._consume_bytes(codestream, 32)
            block_encoded_sizes.append(size)

        last_pos = 0
        bitstreams: list[bitarray] = list()
        for size in block_encoded_sizes:
            start = last_pos
            end = start + size * 8
            bitstreams.append(codestream[start:end])
            last_pos = end

        quantization_matrix = self._get_quantization_matrix(block_size, ndim, quality=quality)

        decoded = np.zeros(shape, dtype=int)
        for bitstream, block in zip(
            bitstreams,
            split_blocks_equal_size(decoded, block_size),
        ):
            mico_decoder = MicoDecoder()
            transformed_block = (
                mico_decoder.decode(
                    bitstream,
                    block.shape,
                )
                * quantization_matrix
            )

            decoded_block: np.ndarray = idctn(transformed_block, norm="ortho")
            decoded_block = decoded_block.round().astype(int)
            block[:] = decoded_block

        return decoded

    def _consume_bytes(self, codestream: bitarray, number_of_bytes: int) -> bitarray:
        value = int.from_bytes(codestream[:number_of_bytes].tobytes())
        codestream[:] = codestream[number_of_bytes:]
        return value

    def _get_quantization_matrix(
        self,
        block_size: int,
        dimensions: int,
        quality: int = 100,
        p: float = 0.8,
    ) -> np.ndarray:
        shape = dimensions * (block_size,)
        aranges = dimensions * (np.arange(1, block_size + 1),)
        quantization_matrix = np.ones(shape)
        for grid in np.meshgrid(*aranges):
            quantization_matrix += grid**p
        quantization_matrix /= dimensions + 1
        quantization_matrix *= quality / 10
        return quantization_matrix
