from typing import Literal

from bitarray import bitarray

from jpig.entropy.probability_model.frequentist_pm import FrequentistPM


class CabacEncoder:
    """
    Implementation of Context Adaptive Binary Arithmetic Coding (CABAC) algorithm.
    Based on https://github.com/ramenhut/abac/blob/master/cabac.cpp
    """

    def __init__(self):
        self.configure_precision(16)
        self.clear()

    def configure_precision(self, precision: int):
        self.entropy_precision = precision
        self._entropy_msb_mask = 1 << (self.entropy_precision - 1)
        self._full_range = (1 << self.entropy_precision) - 1
        self._half_range = self._full_range >> 1
        self._quarter_range = self._half_range >> 1
        self._three_quarter_range = 3 * self._quarter_range
        self.clear()

    def clear(self):
        self.probability_model = FrequentistPM()
        self.result = bitarray()

        self.low = 0
        self.mid = self._half_range
        self.high = self._full_range
        self.e3_count = 0

    def use_model(self, model: FrequentistPM):
        self.probability_model = model

    def encode(self, bits: bitarray, fill_to_byte: bool = False):
        self.start()

        for bit in bits:
            self.encode_bit(bit)

        return self.end(fill_to_byte)

    def start(self, result: bitarray | None = None):
        self.clear()
        if result is not None:
            self.result = result

        return self

    def encode_bit(
        self,
        bit: bool | Literal[0, 1],
        *,
        model: FrequentistPM | None = None,
    ):
        if model is not None:
            self.use_model(model)

        self._update_table()

        if bit:
            self.low = self.mid + 1
            self.probability_model.add_bit(1)
        else:
            self.high = self.mid
            self.probability_model.add_bit(0)

        self._resolve_scaling()

    def end(self, fill_to_byte: bool = False):
        self._flush()
        if fill_to_byte:
            self.result.fill()
        self.result.reverse()
        return self.result

    # Internal use
    def _update_table(self):
        current_range = self.high - self.low
        mid_range = int(current_range * self.probability_model.probability(0))
        self.mid = self.low + mid_range

    def _resolve_scaling(self):
        while True:
            if (self.high & self._entropy_msb_mask) == (
                self.low & self._entropy_msb_mask
            ):
                msb = (self.high & self._entropy_msb_mask) >> (
                    self.entropy_precision - 1
                )
                self.low -= self._half_range * msb + msb
                self.high -= self._half_range * msb + msb

                self.result.append(bool(msb))
                self._flush_inverse_bits(bool(msb))

            elif (self.high <= self._three_quarter_range) and (
                self.low > self._quarter_range
            ):
                self.low -= self._quarter_range + 1
                self.high -= self._quarter_range + 1
                self.e3_count += 1

            else:
                break

            self.low = ((self.low << 1) & self._full_range) | 0
            self.high = ((self.high << 1) & self._full_range) | 1

    def _flush_inverse_bits(self, bit: bool | Literal[0, 1]):
        value = not bool(bit)
        for _ in range(self.e3_count):
            self.result.append(value)
        self.e3_count = 0

    def _flush(self):
        self.e3_count += 1

        if self.low < self._quarter_range:
            self.result.append(0)
            self._flush_inverse_bits(0)
        else:
            self.result.append(1)
            self._flush_inverse_bits(1)
