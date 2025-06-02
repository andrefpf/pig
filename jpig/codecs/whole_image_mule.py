from scipy.fft import dctn, idctn

from jpig.entropy.mule import Mule
from jpig.entropy.cabac import Cabac
from jpig.media.image import RawImage
from jpig.metrics import RD

class WholeImageMule:
    def __init__(self):
        self.estimated_rd = RD()

    def encode(self, image: RawImage, lagrangian: float):
        mule = Mule()
        cabac = Cabac()

        transformed_dct = dctn(image.data, norm="ortho").round().astype(int)
        max_bitplane = mule.find_max_bitplane(transformed_dct)
        mule.encode(transformed_dct, lagrangian, max_bitplane)
        self.estimated_rd = mule.rd
        # print(mule.flags)

        bitstream = []
        bitstream += mule.flags_as_binary()
        bitstream += cabac.encode(mule.signals)
        for bitplane in mule.encoded_bitplanes:
            if bitstream:
                bitstream += cabac.encode(bitplane)
        return bitstream



    def decode(self):
        pass
