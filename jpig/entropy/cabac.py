from typing import Sequence
from collections import deque


class Cabac:
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
        self.frequency_of_zeros = 1
        self.frequency_of_ones = 1
        self.e3_count = 0

        self.low = 0
        self.mid = self._half_range
        self.high = self._full_range

        self.current: int = 0
        self.encoded_data: deque[bool] = deque()
        self.decoded_data: deque[bool] = deque()

    def encode(self, bits: Sequence[bool]):
        self.clear()

        for bit in bits:
            self.encode_bit(bit)

        self.flush()

        return list(self.encoded_data)

    def decode(self, bits: Sequence[bool], size: int):
        self.clear()
        self.encoded_data = deque(bits)
        self.read_first_word()

        for _ in range(size):
            self.decode_bit()

        return list(self.decoded_data)

    def encode_bit(self, bit: bool):
        self.update_table()

        if bit:
            self.low = self.mid + 1
            self.frequency_of_ones += 1
        else:
            self.high = self.mid
            self.frequency_of_zeros += 1

        self.resolve_encoder_scaling()

    def decode_bit(self):
        self.update_table()

        if self.low <= self.current <= self.mid:
            self.high = self.mid
            self.frequency_of_zeros += 1
            self.decoded_data.append(False)

        elif self.mid < self.current <= self.high:
            self.low = self.mid + 1
            self.frequency_of_ones += 1
            self.decoded_data.append(True)

        self.resolve_decoder_scaling()

    def read_first_word(self):
        for _ in range(self.entropy_precision):
            bit = self.encoded_data.popleft()
            self.current = (self.current << 1) | bit

    def update_table(self):
        current_range = self.high - self.low
        encoded_bits = self.frequency_of_zeros + self.frequency_of_ones
        mid_range = current_range * self.frequency_of_zeros // encoded_bits
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

                self.encoded_data.append(bool(msb))
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

            if self.encoded_data:
                bit = self.encoded_data.popleft()

            self.high = ((self.high << 1) & self._full_range) | 1
            self.low = ((self.low << 1) & self._full_range) | 0
            self.current = ((self.current << 1) & self._full_range) | bit

    def flush_inverse_bits(self, bit: bool):
        value = not bool(bit)
        for _ in range(self.e3_count):
            self.encoded_data.append(value)
        self.e3_count = 0

    def flush(self):
        self.e3_count += 1

        if self.low < self._quarter_range:
            self.encoded_data.append(False)
            self.flush_inverse_bits(False)
        else:
            self.encoded_data.append(True)
            self.flush_inverse_bits(True)
