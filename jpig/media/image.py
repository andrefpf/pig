from pathlib import Path

import numpy as np


class RawImage(np.ndarray):
    bitdepth: int

    def __new__(cls, image, **kwargs):
        obj = np.asarray(image, **kwargs).view(cls)
        obj.bitdepth = kwargs.get("bitdepth", 8)
        return obj

    @classmethod
    def from_file(cls, path: str | Path):
        from PIL import Image

        return cls(Image.open(path))

    def width(self):
        if self.ndim < 2:
            return 0
        return self.shape[1]

    def height(self):
        if self.ndim < 2:
            return 0
        return self.shape[0]

    def channels(self):
        if self.ndim < 3:
            return 1
        return self.shape[2]

    def number_of_pixels(self):
        return self.height() * self.width()

    def number_of_samples(self):
        return self.number_of_pixels() * self.channels()

    def get_pixel(self, x: int, y: int) -> np.ndarray:
        return self[y, x]

    def get_sample(self, x: int, y: int, channel: int) -> int:
        return self[y, x, channel]

    def get_channel(self, channel: int) -> np.ndarray:
        return self[:, :, channel]

    def show(self):
        import matplotlib.pyplot as plt

        plt.imshow(
            self,
            vmin=0,
            vmax=(1 << self.bitdepth),
            cmap="gray",
        )
        plt.axis("off")
        plt.show()
