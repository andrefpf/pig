import subprocess
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from jpig.codecs import BlockedMico, BlockedMicoQuantized, BlockedMule
from jpig.media import RawImage
from jpig.metrics.image_metrics import psnr
from jpig.metrics.rate_distortion import RD
from jpig.metrics.rd_curve import find_rd_curve, plot_rd_curves
from jpig.utils.pgx_handler import PGXHandler


def test_jpeg(path: str | Path, quality: int) -> RD:
    path = Path(path).expanduser()
    output_path = Path(".tmp/tmp_jpeg.jpeg")

    original = Image.open(path)
    original.save(output_path, format="jpeg", quality=quality)
    decoded = np.array(Image.open(output_path))

    num_pixels = original.width * original.height
    return RD(
        rate=output_path.stat().st_size * 8 / num_pixels,
        distortion=psnr(original, decoded, n_bits=8),
    )


def test_jpeg_2000(path: str | Path, target_psnr: int) -> RD:
    path = Path(path).expanduser()
    output_path = Path(".tmp/tmp_jpeg_2000.jp2")

    original = Image.open(path)
    original.save(output_path, format="jpeg2000", quality_mode="dB", quality_layers=[target_psnr])
    decoded = np.array(Image.open(output_path))

    num_pixels = original.width * original.height
    return RD(
        rate=output_path.stat().st_size * 8 / num_pixels,
        distortion=psnr(original, decoded, n_bits=8),
    )


def test_webp(path: str, quality: int) -> RD:
    path = Path(path).expanduser()
    output_path = Path(".tmp/tmp_webp.webp")

    original = Image.open(path)
    original.save(output_path, format="webp", quality=quality)
    decoded = np.array(Image.open(output_path).convert("L"))
    num_pixels = original.width * original.height

    return RD(
        rate=output_path.stat().st_size * 8 / num_pixels,
        distortion=psnr(original, decoded, n_bits=8),
    )


def test_jpeg_pleno(path: str, lagrangian: float) -> RD:
    path = Path(path).expanduser()

    Path(".tmp/bla/").mkdir(parents=True, exist_ok=True)

    encoder_command = f"/home/andre/Documents/parallel-jplm/bin/jpl-encoder-bin -i {path} -o .tmp/bla.jpl -c pleno_config.json -errorest"
    encoder_command += f" --lambda {lagrangian} --view_width {625} --view_height {434} --threads 4"
    result = subprocess.run(encoder_command.split())

    decoder_command = "/home/andre/Documents/parallel-jplm/bin/jpl-decoder-bin -i .tmp/bla.jpl -o .tmp/bla/"
    subprocess.run(decoder_command.split())

    
    handler = PGXHandler()
    original = handler.read(path / "0/000_000.pgx")
    decoded = handler.read(".tmp/bla/0/000_000.pgx")
    num_pixels = original.size

    return RD(
        rate=Path(".tmp/bla.jpl").stat().st_size * 8 / num_pixels,
        distortion=psnr(original, decoded, n_bits=10),
    )


def test_mule(path: str, lagrangian: float) -> RD:
    img = RawImage.from_file(path)
    codec_0 = BlockedMule()
    bitstream = codec_0.encode(
        img,
        lagrangian=lagrangian,
        block_size=16,
    )

    codec_1 = BlockedMule()
    decoded = codec_1.decode(bitstream)

    return RD(
        rate=len(bitstream) / img.number_of_samples(),
        distortion=psnr(img, decoded, img.bitdepth),
    )


def test_mico(path: str, lagrangian: float) -> RD:
    img = RawImage.from_file(path)
    codec_0 = BlockedMico()
    bitstream = codec_0.encode(
        img,
        lagrangian=lagrangian,
        block_size=16,
    )

    codec_1 = BlockedMico()
    decoded = codec_1.decode(bitstream)

    return RD(
        rate=len(bitstream) / img.number_of_samples(),
        distortion=psnr(img, decoded, img.bitdepth),
    )


def test_mico_quantized(path: str, quality: int) -> RD:
    img = RawImage.from_file(path)
    codec_0 = BlockedMicoQuantized()
    bitstream = codec_0.encode(
        img,
        lagrangian=1e-8,
        quality=quality,
        block_size=16,
    )

    codec_1 = BlockedMicoQuantized()
    decoded = codec_1.decode(bitstream)

    return RD(
        rate=len(bitstream) / img.number_of_samples(),
        distortion=psnr(img, decoded, img.bitdepth),
    )


if __name__ == "__main__":
    path = Path("datasets/images/bikes_middle_view.pgm").expanduser()
    path_pleno = Path("datasets/images/Bikes/").expanduser()

    parameters = [
        (10,),
        # (20,),
        # (30,),
        # (40,),
        (50,),
        # (60,),
        # (70,),
        (80,),
        (90,),
        # (100,),
    ]

    jpeg_2k_parameters = [
        (30,),
        (35,),
        (37,),
        (40,),
    ]

    lagrangians_mule = [
        (10,),
        (100,),
        (500,),
        (1000,),
    ]

    lagrangians_mico = [
        (10,),
        (100,),
        (500,),
        (1000,),
    ]

    lagrangians_pleno = [
        (10,),
        (100,),
        (500,),
        (1000,),
    ]

    plot_rd_curves(
        webp_curve=find_rd_curve(partial(test_webp, path), parameters),
        jpeg_curve=find_rd_curve(partial(test_jpeg, path), parameters),
        jpeg_2000_curve=find_rd_curve(partial(test_jpeg_2000, path), jpeg_2k_parameters),
        jpeg_pleno_curve=find_rd_curve(partial(test_jpeg_pleno, path_pleno), lagrangians_pleno),
        mule_curve=find_rd_curve(partial(test_mule, path), lagrangians_mule),
        mico_curve=find_rd_curve(partial(test_mico, path), lagrangians_mico),
        mico_quantized_curve=find_rd_curve(partial(test_mico_quantized, path), parameters),
    )
