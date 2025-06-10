import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacEncoder, FrequentistPM
from jpig.metrics import RD
from jpig.utils.block_utils import bigger_possible_slice, split_shape_in_half


class MicoEncoder:
    """
    Multidimensional Image COdec - Encoder
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.estimated_rd = RD()
        self.flags = ""
        self.block = np.array([])
        self.bitplane_sizes = []

        self.upper_bitplane = 32
        self.lagrangian = 10_000

        self.flags_model = FrequentistPM()
        self.signals_model = FrequentistPM()
        self.bitplane_sizes_model = FrequentistPM()
        self.bitplane_models = [FrequentistPM() for _ in range(32)]

        self.bitstream = bitarray()
        self.cabac = CabacEncoder()

    def encode(
        self,
        block: np.ndarray,
        lagrangian: float = 10_000,
        *,
        upper_bitplane: int = 32,
    ) -> bitarray:
        self.clear()

        self.block = block
        self.lagrangian = lagrangian
        self.upper_bitplane = upper_bitplane

        self.bitplane_sizes = self._calculate_bitplane_sizes()
        # self._encode_bitplane_sizes()

        self.flags, _ = self._recursive_optimize_encoding_tree(
            bigger_possible_slice(block.shape)
        )
        self._clear_models()

        self.cabac.start(result=self.bitstream)
        self.apply_encoding(list(self.flags), bigger_possible_slice(block.shape))
        return self.cabac.end(fill_to_byte=True)

    def apply_encoding(self, flags: list[str], block_position: tuple[slice]):
        flag = flags.pop(0)

        if flag not in ["Z", "C"]:
            raise ValueError("Invalid encoding")

        if flag == "Z":
            self.cabac.encode_bit(0, model=self.flags_model)
            return

        self.cabac.encode_bit(1, model=self.flags_model)
        sub_block = self.block[block_position]
        if sub_block.size > 1:
            for sub_pos in split_shape_in_half(block_position):
                self.apply_encoding(flags, sub_pos)
            return

        bitplane = self._get_bitplane(block_position)
        value = sub_block.flatten()[0]
        for i in range(0, bitplane):
            bit = (1 << i) & np.abs(value) != 0
            self.cabac.encode_bit(bit, model=self.bitplane_models[i])
        self.cabac.encode_bit(value < 0, model=self.signals_model)

    def _recursive_optimize_encoding_tree(
        self, block_position: tuple[slice]
    ) -> tuple[str, float]:
        sub_block = self.block[block_position]

        zero_rd = RD(
            rate=1,
            distortion=np.sum(sub_block**2),
        )

        if zero_rd.distortion == 0:
            self.flags_model.add_bit(0)
            return "Z", 0

        if sub_block.size == 1:
            value = sub_block.flatten()[0]
            bitplane = self._get_bitplane(block_position)
            for i in range(bitplane):
                model = self.bitplane_models[i]
                model.add_bit((1 << i) & np.abs(value) != 0)
            return "C", 0

        self._push_models()
        continue_rd = RD(1, 0)
        continue_flags = "C"
        self.flags_model.add_bit(1)

        for sub_pos in split_shape_in_half(block_position):
            current_flags, distortion = self._recursive_optimize_encoding_tree(sub_pos)
            continue_flags += current_flags
            continue_rd.distortion += distortion
        continue_rd.rate = self._estimate_current_rate()

        continue_cost = continue_rd.cost(self.lagrangian / sub_block.size)
        zero_cost = zero_rd.cost(self.lagrangian / sub_block.size)

        if continue_cost < zero_cost:
            return continue_flags, continue_rd.distortion
        else:
            self._pop_models()
            self.flags_model.add_bit(0)
            return "Z", 0

    def _encode_bitplane_sizes(self):
        last_size = 0
        for size in reversed(self.bitplane_sizes):
            difference = size - last_size
            for _ in range(difference):
                self.cabac.encode_bit(1, model=self.bitplane_sizes_model)
            self.cabac.encode_bit(0, model=self.bitplane_sizes_model)
            last_size = size

    def _calculate_bitplane_sizes(self):
        tmp_block = self.block.copy()
        bitplane_sizes = []

        for i in range(max(self.block.shape)):
            slices = tuple(slice(0, i) for _ in range(self.block.ndim))
            tmp_block[*slices] = 0

            bp = self.find_max_bitplane(tmp_block)
            bitplane_sizes.append(bp)

        return bitplane_sizes

    def _get_bitplane(self, block_position: tuple[slice]):
        level = max(s.start for s in block_position)
        return self.bitplane_sizes[level]

    def _push_models(self):
        for model in self.probability_models():
            model.push()

    def _pop_models(self):
        for model in self.probability_models():
            model.pop()

    def _clear_models(self):
        for model in self.probability_models():
            model.clear()

    def _estimate_current_rate(self) -> float:
        total_size = 0
        for model in self.probability_models():
            total_size += model.estimated_rate()
        return total_size

    def probability_models(self):
        return [
            self.flags_model,
            self.signals_model,
            self.bitplane_sizes_model,
            *self.bitplane_models,
        ]

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()
