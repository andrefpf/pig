import numpy as np
from pathlib import Path


class JPIGImage:
    def __init__(self) -> None:
        self.data = np.array()
    
    def width(self):
        pass

    def height(self):
        pass

    def get_pixel(self, x: int, y: int) -> tuple[int, int, int]:
        return (0, 0, 0)  # rgb

    def get_sample(self, x: int, y: int, channel: int) -> int:
        return 0

    def show(self):
        pass

    def load_from_png(self, path: str | Path):
        path = Path
        # implement the loader here

    def load_from_ppm(self, path: str | Path):
        path = Path
        # implement the loader here

    def load_from_bitmap(self, path: str | Path):
        path = Path
        # implement the loader here

    def load_from_jpeg(self, path: str | Path):
        path = Path
        # implement the loader here