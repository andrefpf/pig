import numpy as np
from bitarray import bitarray
from scipy.fft import dctn, idctn

from pig.entropy import MicoDecoder, MicoEncoder


class WholeImageMico:
    def encode(self, data: np.ndarray, lagrangian: float) -> bitarray:
        transformed: np.ndarray = dctn(data, norm="ortho")
        transformed = transformed.round().astype(int)

        mico_encoder = MicoEncoder()

        header = bitarray()
        header.extend(f"{data.ndim:08b}")
        for size in data.shape:
            header.extend(f"{size:032b}")

        bitstream = mico_encoder.encode(
            transformed,
            lagrangian,
        )
        return header + bitstream

    def decode(self, codestream: bitarray) -> np.ndarray:
        codestream = codestream.copy()
        mico_decoder = MicoDecoder()

        ndim = self._consume_bytes(codestream, 8)
        shape = list()
        for _ in range(ndim):
            size = self._consume_bytes(codestream, 32)
            shape.append(size)

        transformed_decoded = mico_decoder.decode(
            codestream,
            shape,
        )

        decoded: np.ndarray = idctn(transformed_decoded, norm="ortho")
        decoded = decoded.round().astype(int)
        return decoded

    def _consume_bytes(self, codestream: bitarray, number_of_bytes: int) -> bitarray:
        value = int.from_bytes(codestream[:number_of_bytes].tobytes())
        codestream[:] = codestream[number_of_bytes:]
        return value
