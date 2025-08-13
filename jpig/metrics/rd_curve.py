from itertools import zip_longest
from typing import Callable, Sequence

import matplotlib.pyplot as plt
from tqdm import tqdm

from jpig.metrics import RD


def find_rd_curve(
    function: Callable[..., RD],
    args_sequence: Sequence[tuple] | None = None,
    kwargs_sequence: Sequence[dict] | None = None,
) -> list[RD]:
    if args_sequence is None:
        args_sequence = list()

    if kwargs_sequence is None:
        kwargs_sequence = list()

    rd_curve: list[RD] = list()

    for args, kwargs in tqdm(
        zip_longest(args_sequence, kwargs_sequence, fillvalue=None),
        total=max(
            len(args_sequence),
            len(kwargs_sequence),
        ),
    ):
        if args is None:
            args = tuple()

        if kwargs is None:
            kwargs = dict()

        rd_curve.append(function(*args, **kwargs))

    rd_curve.sort(key=lambda rd: rd.rate)

    return rd_curve


def plot_rd_curves(**all_rd_curves: Sequence[RD]):
    for name, curve in all_rd_curves.items():
        r = [c.rate for c in curve]
        d = [c.distortion for c in curve]
        plt.plot(r, d, marker="o", label=name)

    plt.xlabel("BPP")
    plt.ylabel("PSNR")

    plt.legend()
    plt.show()


def get_bdrate(reference: Sequence[RD], test: Sequence[RD]) -> float:
    return 0
