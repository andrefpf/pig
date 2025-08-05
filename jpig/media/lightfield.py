from pathlib import Path

import numpy as np


class RawLightField(np.ndarray):
    bitdepth: int

    def __new__(cls, lightfield, **kwargs):
        bitdepth = kwargs.pop("bitdepth", 8)
        obj = np.asarray(lightfield, **kwargs).view(cls)
        obj.bitdepth = bitdepth
        return obj

    @classmethod
    def from_file(cls, path: str | Path):
        import re

        from jpig.utils.pgx_handler import PGXHandler

        t_view_regex = re.compile(r"[0-9]+(?=_)")
        s_view_regex = re.compile(r"(?<=_)[0-9]+")

        path = Path(path).expanduser()
        reader = PGXHandler()

        n_channels = 0
        t_size = 0
        s_size = 0
        v_size = 0
        u_size = 0
        bitdepth = 0

        max_channel = max(path.glob("*"))
        max_view_path = max(max_channel.glob("*"))
        with open(max_view_path, "rb") as file:
            max_view_header = reader._read_header(file)
            bitdepth = max_view_header.depth

        n_channels = int(max_channel.name) + 1

        try:
            t_size = int(t_view_regex.search(max_view_path.stem).group())
            s_size = int(s_view_regex.search(max_view_path.stem).group())
        except Exception as e:
            raise ValueError("Invalid light field name") from e

        v_size = max_view_header.height
        u_size = max_view_header.width
        t_size += 1
        s_size += 1

        data = np.empty((t_size, s_size, v_size, u_size, n_channels), dtype=int)
        for c in range(n_channels):
            for t in range(t_size):
                for s in range(s_size):
                    view = reader.read(path / f"{c}/{t:03}_{s:03}.pgx")
                    data[t, s, :, :, c] = view

        return cls(data, bitdepth=bitdepth)

    def channels(self):
        if self.ndim < 3:
            return 1
        return self.shape[4]

    def t(self):
        return self.shape[0]

    def s(self):
        return self.shape[1]

    def v(self):
        return self.shape[2]

    def u(self):
        return self.shape[3]

    def number_of_pixels(self):
        return self.t() * self.s() * self.v() * self.u()

    def number_of_samples(self):
        return self.number_of_pixels() * self.channels()

    def get_pixel(self, t: int, s: int, v: int, u: int) -> np.ndarray:
        return self[t, s, v, u]

    def get_sample(self, t: int, s: int, v: int, u: int, channel: int) -> int:
        return self[t, s, v, u, channel]

    def get_view(self, t: int, s: int) -> np.ndarray:
        return self[t, s]

    def get_channel(self, channel: int) -> np.ndarray:
        return self[:, :, :, :, channel]
