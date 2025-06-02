from scipy.fft import dctn, idctn
import numpy as np

from jpig.entropy.mule import Mule
from jpig.entropy.cabac import Cabac
from jpig.media.image import RawImage
from jpig.utils.block_utils import split_blocks_equal_size
from jpig.metrics import RD


class BlockedMule:
    def __init__(self):
        self.estimated_rd = RD()

    def encode(self, image: RawImage, lagrangian: float = 10_000, block_size: int = 16):
        cabac = Cabac()
        bitstream = []

        number_of_blocks = (max(image.data.shape) // block_size) ** 2
        number_of_values = block_size ** image.data.ndim
        maximum_representable_number = 1 << image.bitdepth
        max_bitplane = (number_of_values * maximum_representable_number).bit_length()

        for i, block in enumerate(split_blocks_equal_size(image.data, block_size)):
            mule = Mule()
            transformed_dct = dctn(block, norm="ortho").round().astype(int)
            mule.encode(transformed_dct, lagrangian, max_bitplane)
            self.estimated_rd += mule.rd

            bitstream += mule.flags_as_binary()
            bitstream += cabac.encode(mule.signals)
            for bitplane in mule.encoded_bitplanes:
                if bitstream:
                    bitstream += cabac.encode(bitplane)

        return bitstream

    def decode(self):
        pass
