import numpy as np
from pathlib import Path
from PIL import Image


class RawImage:
    def __init__(self) -> None:
        self.data = np.array([])

    def width(self):
        if len(self.data.shape) < 2:
            return 0
        return self.data.shape[1]

    def height(self):
        if len(self.data.shape) < 2:
            return 0
        return self.data.shape[0]

    def channels(self):
        if len(self.data.shape) < 3:
            return 1
        return self.data.shape[2]

    def get_pixel(self, x: int, y: int) -> np.ndarray:
        return self.data[y, x]

    def get_sample(self, x: int, y: int, channel: int) -> int:
        return self.data[y, x, channel]

    def get_channel(self, channel: int) -> np.ndarray:
        return self.data[:, :, channel]

    def load_file(self, path: str | Path):
        self.data = np.array(Image.open(path))
        return self

    def show(self):
        import matplotlib.pyplot as plt

        plt.imshow(self.data, cmap="gray")
        plt.show()
