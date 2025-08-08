from typing import Callable, Sequence

import matplotlib.pyplot as plt

from jpig.metrics import RD


def find_rd_curve(
    function: Callable[..., RD],
    args_sequence: Sequence[tuple] | None = None,
    kwargs_sequence: Sequence[dict] | None = None,
) -> list[RD]:
    if args_sequence is None:
        args_sequence = tuple()

    if kwargs_sequence is None:
        kwargs_sequence = dict()

    rd_curve: list[RD] = list()

    for args, kwargs in zip(args_sequence, kwargs_sequence):
        rd_curve.append(function(*args, **kwargs))

    rd_curve.sort(key=lambda rd: rd.rate)

    return rd_curve


def plot_rd_curves(*all_rd_curves: Sequence[RD]):
    for curve in all_rd_curves:
        r = [c.rate for c in curve]
        d = [c.distortion for c in curve]
        plt.plot(r, d)

    plt.show()


def get_bdrate(reference: Sequence[RD], test: Sequence[RD]) -> float:
    return 0
