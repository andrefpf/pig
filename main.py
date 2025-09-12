import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn

from jpig.codecs import BlockedMicoQuantized, BlockedMule
from jpig.entropy.mico.mico_decoder import MicoDecoder
from jpig.entropy.mico.mico_encoder import MicoEncoder
from jpig.entropy.mico.mico_optimizer import MicoOptimizer
from jpig.entropy.mule.mule_optimizer import MuleOptimizer
from jpig.media import RawImage
from jpig.metrics import mse, psnr
from jpig.utils.block_utils import bigger_possible_slice


def compare_data(data1, data2, bitdepth=8):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(
        data1,
        vmin=0,
        vmax=(1 << bitdepth),
        cmap="gray",
    )
    axes[0].axis("off")
    axes[0].set_title("Image 1")

    axes[1].imshow(
        data2,
        vmin=0,
        vmax=(1 << bitdepth),
        cmap="gray",
    )
    axes[1].axis("off")
    axes[1].set_title("Image 2")

    plt.show()


def main():
    image = RawImage.from_file("./datasets/images/Bikes_64x64/0/000_000.pgx")[16:32, 16:32]
    frequency = dctn(image, norm="ortho").astype(int)

    encoder = MicoEncoder()
    encoded = encoder.encode(frequency, 1_000)

    decoder = MicoDecoder()
    decoded = decoder.decode(encoded, frequency.shape)
    reconstructed = idctn(decoded, norm="ortho").astype(int)
    
    print()
    print(f"PSNR: {psnr(image, reconstructed, 10)}")
    print(f"MSE: {mse(image, reconstructed)}")
    print(f"BPP:", len(encoded) / image.size)

    compare_data(image, reconstructed, 10)


if __name__ == "__main__":
    main()
