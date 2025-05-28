import numpy as np
from jpig.entropy.mule import Mule


def test_mule():
    matrix = np.array(
        [
            [12, 8, 0, 2],
            [-7, 3, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 3, -2],
        ]
    )

    # Encoding with lagrangian equals to zero
    # i.e. without introducing losses
    mule = Mule()
    mule.encode(matrix, 0, 4)
    # print(mule)
