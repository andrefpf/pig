import numpy as np
from bitarray import bitarray
from scipy.fft import dctn, idctn

from pig.entropy import MuleDecoder, MuleEncoder


class WholeImageMule:
    def encode(self, data: np.ndarray, lagrangian: float) -> bitarray:
        transformed: np.ndarray = dctn(data, norm="ortho")
        transformed = transformed.round().astype(int)

        max_bitplane = MuleEncoder.find_max_bitplane(transformed)
        mule_encoder = MuleEncoder()

        header = bitarray()
        header.extend(f"{data.ndim:08b}")
        for size in data.shape:
            header.extend(f"{size:032b}")
        header.extend(f"{max_bitplane:08b}")

        bitstream = mule_encoder.encode(
            transformed,
            lagrangian,
            upper_bitplane=max_bitplane,
        )
        return header + bitstream

    def decode(self, codestream: bitarray) -> np.ndarray:
        codestream = codestream.copy()
        mule_decoder = MuleDecoder()

        ndim = self._consume_bytes(codestream, 8)
        shape = list()
        for _ in range(ndim):
            size = self._consume_bytes(codestream, 32)
            shape.append(size)

        max_bitplane = self._consume_bytes(codestream, 8)
        transformed_decoded = mule_decoder.decode(
            codestream,
            shape,
            upper_bitplane=max_bitplane,
        )

        decoded: np.ndarray = idctn(transformed_decoded, norm="ortho")
        decoded = decoded.round().astype(int)
        return decoded

    def _consume_bytes(self, codestream: bitarray, number_of_bytes: int) -> bitarray:
        value = int.from_bytes(codestream[:number_of_bytes].tobytes())
        codestream[:] = codestream[number_of_bytes:]
        return value
