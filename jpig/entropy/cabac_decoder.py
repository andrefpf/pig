from bitarray import bitarray

from jpig.entropy.probability_model import ProbabilityModel


class CabacDecoder:
    """
    Implementation of Context Adaptive Binary Arithmetic Coding (CABAC) algorithm.
    Based on https://github.com/ramenhut/abac/blob/master/cabac.cpp
    """

    def __init__(self):
        self.configure_precision(16)

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
        self.current: int = 0
        self.read_bits: int = 0

        self.buffer = bitarray()
        self.result = bitarray()

        self.low = 0
        self.mid = self._half_range
        self.high = self._full_range

    def use_model(self, model: ProbabilityModel):
        self.probability_model = model

    def decode(self, bits: bitarray, size: int):
        self.start(bits.copy())

        for _ in range(size):
            self.decode_bit()

        return self.end()

    def start(self, bits: bitarray, result: bitarray | None = None):
        self.clear()
        if result is not None:
            self.result = result

        self.buffer = bits.copy()
        self._read_first_word()
        return self

    def decode_bit(self, *, model: ProbabilityModel | None = None):
        if model is not None:
            self.use_model(model)

        output = None
        self._update_table()

        if self.low <= self.current <= self.mid:
            self.high = self.mid
            self.probability_model.add_bit(0)
            self.result.append(0)
            output = 0

        elif self.mid < self.current <= self.high:
            self.low = self.mid + 1
            self.probability_model.add_bit(1)
            self.result.append(1)
            output = 1

        else:
            raise ValueError("Invalid encoding sequence.")

        self._resolve_scaling()
        return output

    def end(self):
        return self.result

    # Internal use
    def _read_first_word(self):
        for _ in range(self.entropy_precision):
            if self.buffer:
                bit = self.buffer.pop()
            self.current = (self.current << 1) | bit
            self.read_bits += 1

    def _update_table(self):
        current_range = self.high - self.low
        mid_range = int(current_range * self.probability_model.probability(0))
        self.mid = self.low + mid_range

    def _resolve_scaling(self):
        bit = 0

        while True:
            if self.high <= self._half_range:
                pass

            elif self._half_range < self.low:
                self.high -= self._half_range + 1
                self.low -= self._half_range + 1
                self.current -= self._half_range + 1

            elif (self._quarter_range < self.low) and (self.high <= self._three_quarter_range):
                self.high -= self._quarter_range + 1
                self.low -= self._quarter_range + 1
                self.current -= self._quarter_range + 1

            else:
                return

            if self.buffer:
                bit = self.buffer.pop()
                self.read_bits += 1

            self.high = ((self.high << 1) & self._full_range) | 1
            self.low = ((self.low << 1) & self._full_range) | 0
            self.current = ((self.current << 1) & self._full_range) | bit
