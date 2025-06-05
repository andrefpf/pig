import numpy as np
from scipy.fft import dctn, idctn
from jpig.entropy import MuleEncoder, MuleDecoder
from bitarray import bitarray
from jpig.utils.block_utils import split_blocks_equal_size
from itertools import pairwise


class BlockedMule:
    def encode(self, data: np.ndarray, lagrangian: float, block_size: int = 8) -> bitarray:

        max_bitplane = 32

        bitstream = bitarray()
        block_encoded_sizes = []
        for block in split_blocks_equal_size(data, block_size):
            mule_encoder = MuleEncoder()
            transformed_block: np.ndarray = dctn(block, norm="ortho")
            transformed_block = transformed_block.round().astype(int)
            block_bitstream = mule_encoder.encode(transformed_block, lagrangian, max_bitplane)
            bitstream += block_bitstream
            block_encoded_sizes.append(len(block_bitstream) // 8)

        header = bitarray()
        header.extend(f"{data.ndim:08b}")
        for size in data.shape:
            header.extend(f"{size:032b}")
        header.extend(f"{max_bitplane:08b}")

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

        max_bitplane = self._consume_bytes(codestream, 8)
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

        decoded = np.zeros(shape, dtype=int)
        for bitstream, block in zip(bitstreams, split_blocks_equal_size(decoded, block_size)):
            mule_decoder = MuleDecoder()
            transformed_block = mule_decoder.decode(bitstream, block.shape, max_bitplane)
            decoded_block: np.ndarray = idctn(transformed_block, norm="ortho")
            decoded_block = decoded_block.round().astype(int)
            block[:] = decoded_block

        return decoded

    def _consume_bytes(self, codestream: bitarray, number_of_bytes: int) -> bitarray:
        value = int.from_bytes(codestream[:number_of_bytes].tobytes())
        codestream[:] = codestream[number_of_bytes:]
        return value
