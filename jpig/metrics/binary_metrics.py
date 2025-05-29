from typing import Sequence
import numpy as np


def binary_entropy(sequence: Sequence[bool]) -> float:
    number_of_bits = len(sequence)
    if number_of_bits == 0:
        return 0
    
    number_of_1 = sum(sequence)
    if number_of_1 == 0:
        return 0

    number_of_0 = number_of_bits - number_of_1
    if number_of_0 == 0:
        return 0
    
    prob_0 = number_of_0 / number_of_bits
    prob_1 = number_of_1 / number_of_bits

    return -prob_0 * np.log2(prob_0) - prob_1 * np.log2(prob_1)