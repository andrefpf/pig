import numpy as np

from jpig.entropy import MuleEncoder, MuleDecoder
from jpig.media import RawImage
from jpig.metrics import mse

from scipy.fft import dctn, idctn

import matplotlib.pyplot as plt


def main():
    img = RawImage().load_file("./datasets/images/cameraman.pgm")
    transformed = dctn(img.data, norm="ortho").astype(int)

    max_bitplane = MuleEncoder.find_max_bitplane(transformed)
    mule_encoder = MuleEncoder()
    bitstream = mule_encoder.encode(transformed, 100, max_bitplane)

    mule_decoder = MuleDecoder()
    transformed_decoded = mule_decoder.decode(mule_encoder.bitstream, transformed.shape, max_bitplane)

    decoded = idctn(transformed_decoded, norm="ortho")

    print(f"Original Rate: {img.number_of_samples() * img.bitdepth / 8000:.2f} Kb")
    print(f"Actual Rate: {len(bitstream) / 8000:.2f} Kb")
    print(f"Estimated Rate: {mule_encoder.estimated_rd.rate / 8000:.2f} Kb")
    print(f"MSE: {mse(img.data, decoded)}")
    print(f"Estimated MSE: {mule_encoder.estimated_rd.distortion / img.number_of_samples()}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(
        img.data,
        vmin=0,
        vmax=(1 << img.bitdepth),
        cmap="gray",
    )
    axes[0].axis("off")
    axes[0].set_title("Image 1")

    axes[1].imshow(
        decoded,
        vmin=0,
        vmax=(1 << img.bitdepth),
        cmap="gray",
    )
    axes[1].axis("off")
    axes[1].set_title("Image 2")

    plt.show()


if __name__ == "__main__":
    main()
