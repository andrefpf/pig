from jpig.media import RawImage
from jpig.metrics import mse, psnr
from jpig.codecs import BlockedMule


import matplotlib.pyplot as plt


def compare_data(data1, data2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(
        data1,
        vmin=0,
        vmax=(1 << 8),
        cmap="gray",
    )
    axes[0].axis("off")
    axes[0].set_title("Image 1")

    axes[1].imshow(
        data2,
        vmin=0,
        vmax=(1 << 8),
        cmap="gray",
    )
    axes[1].axis("off")
    axes[1].set_title("Image 2")

    plt.show()


def main():
    img = RawImage().load_file("./datasets/images/cameraman.pgm")

    codec = BlockedMule()
    bitstream = codec.encode(img.data, 0, 8)
    decoded = codec.decode(bitstream)

    print(f"Original Rate: {img.number_of_samples() * img.bitdepth / 8000:.2f} Kb")
    print(f"Rate: {len(bitstream) / 8000:.2f} Kb")
    print(f"MSE: {mse(img.data, decoded)}")
    print(f"PSNR: {psnr(img.data, decoded, img.bitdepth)}")

    compare_data(img.data, decoded)


if __name__ == "__main__":
    main()
