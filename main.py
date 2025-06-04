import numpy as np

from jpig.entropy import MuleEncoder, MuleDecoder
from jpig.media import RawImage
from jpig.metrics import mse

from scipy.fft import dctn, idctn

import matplotlib.pyplot as plt 


def main():
    img = RawImage().load_file("./datasets/images/cameraman.bmp")
    img.show()

    max_bitplane = MuleEncoder.find_max_bitplane(img.data)
    mule_encoder = MuleEncoder()
    bitstream = mule_encoder.encode(img.data, 0, max_bitplane)

    rate = len(bitstream)
    print(f"Actual Rate: {rate / 8000} Kb")
    print(f"Estimated Rate: {mule_encoder.estimated_rd.rate / 8000} Kb")

    mule_decoder = MuleDecoder()
    block = mule_decoder.decode(mule_encoder.bitstream, img.data.shape, max_bitplane)

    distortion = mse(img.data, block)
    print(f"MSE: {distortion}")
    print(f"Estimated MSE: {mule_encoder.estimated_rd.rate / img.number_of_samples()}")

    plt.imshow(block, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
