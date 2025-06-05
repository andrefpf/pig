import numpy as np
from scipy.fft import dctn, idctn
from jpig.entropy import MuleEncoder, MuleDecoder
from bitarray import bitarray


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

        bitstream = mule_encoder.encode(transformed, lagrangian, max_bitplane)
        return header + bitstream

    def decode(self, codestream: bitarray) -> np.ndarray:
        codestream = codestream.copy()
        
        ndim, codestream = int.from_bytes(codestream[:8].tobytes()), codestream[8:]
        shape = list()
        for _ in range(ndim):
            size, codestream = int.from_bytes(codestream[:32].tobytes()), codestream[32:]
            shape.append(size)

        max_bitplane, codestream = int.from_bytes(codestream[:8].tobytes()), codestream[32:]

        mule_decoder = MuleDecoder()
        transformed_decoded = mule_decoder.decode(codestream, shape, max_bitplane)

        decoded: np.ndarray = idctn(transformed_decoded, norm="ortho")
        decoded = decoded.round().astype(int)
        return decoded
