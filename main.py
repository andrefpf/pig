import matplotlib.pyplot as plt

from jpig.codecs import BlockedMicoQuantized, BlockedMule
from jpig.media import RawImage
from jpig.metrics import mse, psnr


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
    img = RawImage.from_file("./datasets/images/cameraman.pgm")
    block_size = 8
    
    codec = BlockedMicoQuantized()
    
    from time import perf_counter
    start = perf_counter()
    bitstream = codec.encode(img, 1e-8, 90, block_size)
    end = perf_counter()
    print("Time: ", end - start)


    decoded = codec.decode(bitstream)

    print(f"Original Rate: {img.number_of_samples() * img.bitdepth / 8000:.2f} Kb")
    print(f"Rate: {len(bitstream) / 8000:.2f} Kb")
    print(f"MSE: {mse(img, decoded)}")
    print(f"PSNR: {psnr(img, decoded, img.bitdepth)}")
    # compare_data(img, decoded)

    print("-" * 10)

    codec = BlockedMule()
    bitstream = codec.encode(img, 40, block_size)
    decoded = codec.decode(bitstream)

    print(f"Original Rate: {img.number_of_samples() * img.bitdepth / 8000:.2f} Kb")
    print(f"Rate: {len(bitstream) / 8000:.2f} Kb")
    print(f"MSE: {mse(img, decoded)}")
    print(f"PSNR: {psnr(img, decoded, img.bitdepth)}")
    # compare_data(img, decoded)


if __name__ == "__main__":
    main()
