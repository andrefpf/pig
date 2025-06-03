from typing import Literal
from bitarray import bitarray
from jpig.entropy.probability_model import ProbabilityModel


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
        self.probability_model = ProbabilityModel()
        # self.probability_model.frequency(0) = 1
        # self.probability_model.frequency(1) = 1
        self.e3_count = 0

        self.low = 0
        self.mid = self._half_range
        self.high = self._full_range

        self.current: int = 0
        self._buffer = bitarray()
        self._result = bitarray()

    def encode(self, bits: bitarray):
        self.clear()

        for bit in bits:
            self.encode_bit(bit)

        self.flush()

        return self._result

    def decode(self, bits: bitarray, size: int):
        self.clear()
        self._buffer = bits.copy()
        self._buffer.reverse()
        self.read_first_word()

        for _ in range(size):
            self.decode_bit()

        return self._result

    def encode_bit(self, bit: bool):
        self.update_table()

        if bit:
            self.low = self.mid + 1
            self.probability_model.add_bit(1)
        else:
            self.high = self.mid
            self.probability_model.add_bit(0)

        self.resolve_encoder_scaling()

    def decode_bit(self):
        self.update_table()

        if self.low <= self.current <= self.mid:
            self.high = self.mid
            self.probability_model.add_bit(0)
            self._result.append(0)

        elif self.mid < self.current <= self.high:
            self.low = self.mid + 1
            self.probability_model.add_bit(1)
            self._result.append(1)

        self.resolve_decoder_scaling()

    def read_first_word(self):
        for _ in range(self.entropy_precision):
            bit = self._buffer.pop()
            self.current = (self.current << 1) | bit

    def update_table(self):
        current_range = self.high - self.low
        mid_range = int(current_range * self.probability_model.probability(0))
        self.mid = self.low + mid_range

    def resolve_encoder_scaling(self):
        while True:
            if (self.high & self._entropy_msb_mask) == (
                self.low & self._entropy_msb_mask
            ):
                msb = (self.high & self._entropy_msb_mask) >> (
                    self.entropy_precision - 1
                )
                self.low -= self._half_range * msb + msb
                self.high -= self._half_range * msb + msb

                self._result.append(bool(msb))
                self.flush_inverse_bits(bool(msb))

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

    def resolve_decoder_scaling(self):
        bit = 0

        while True:
            if self.high <= self._half_range:
                pass

            elif self._half_range < self.low:
                self.high -= self._half_range + 1
                self.low -= self._half_range + 1
                self.current -= self._half_range + 1

            elif (self._quarter_range < self.low) and (
                self.high <= self._three_quarter_range
            ):
                self.high -= self._quarter_range + 1
                self.low -= self._quarter_range + 1
                self.current -= self._quarter_range + 1

            else:
                return

            if self._buffer:
                bit = self._buffer.pop()

            self.high = ((self.high << 1) & self._full_range) | 1
            self.low = ((self.low << 1) & self._full_range) | 0
            self.current = ((self.current << 1) & self._full_range) | bit

    def flush_inverse_bits(self, bit: bool | Literal[0, 1]):
        value = not bool(bit)
        for _ in range(self.e3_count):
            self._result.append(value)
        self.e3_count = 0

    def flush(self):
        self.e3_count += 1

        if self.low < self._quarter_range:
            self._result.append(0)
            self.flush_inverse_bits(0)
        else:
            self._result.append(1)
            self.flush_inverse_bits(1)
