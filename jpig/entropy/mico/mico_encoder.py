import numpy as np
from bitarray import bitarray

from jpig.entropy import CabacEncoder, FrequentistPM
from jpig.metrics import RD, energy
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
        self.lagrangian = 10_000

        self.split_flags_model = FrequentistPM()
        self.unit_flags_model = FrequentistPM()
        self.block_flags_model = FrequentistPM()

        self.signals_model = FrequentistPM()
        self.bitplane_models = [FrequentistPM() for _ in range(32)]
        self.bitplane_sizes_model = FrequentistPM()

        self.bitstream = bitarray()
        self.cabac = CabacEncoder()

    def encode(
        self,
        block: np.ndarray,
        lagrangian: float = 10_000,
    ) -> bitarray:
        self.clear()

        self.block = block
        self.lagrangian = lagrangian

        self.bitplane_sizes = self._calculate_bitplane_sizes()
        self.flags, self.estimated_rd = self._recursive_optimize_encoding_tree(
            bigger_possible_slice(block.shape)
        )
        self._clear_models()

        self.cabac.start(result=self.bitstream)
        self.encode_bitplane_sizes()
        self.apply_encoding(list(self.flags), bigger_possible_slice(block.shape))
        return self.cabac.end(fill_to_byte=True)

    def encode_bitplane_sizes(self):
        last_size = 0
        for size in reversed(self.bitplane_sizes):
            difference = size - last_size
            for _ in range(difference):
                self.cabac.encode_bit(1, model=self.bitplane_sizes_model)
            self.cabac.encode_bit(0, model=self.bitplane_sizes_model)
            last_size = size

    def apply_encoding(self, flags: list[str], block_position: tuple[slice]):
        flag = flags.pop(0)
        sub_block = self.block[block_position]

        if flag == "S":  # Split
            self.cabac.encode_bit(1, model=self.split_flags_model)

            for sub_pos in split_shape_in_half(block_position):
                self.apply_encoding(flags, sub_pos)

        elif flag == "F":  # Full
            self.cabac.encode_bit(0, model=self.split_flags_model)
            self.cabac.encode_bit(1, model=self.block_flags_model)

            dims = np.mgrid[block_position]
            levels: np.ndarray = np.max(dims, axis=0)
            for level, value in zip(levels.flatten(), sub_block.flatten()):
                upper_bitplane = self.bitplane_sizes[level]
                self.encode_value(value, upper_bitplane)

        elif flag == "E":  # Empty
            self.cabac.encode_bit(0, model=self.split_flags_model)
            self.cabac.encode_bit(0, model=self.block_flags_model)

        elif flag == "v":  # Value
            assert sub_block.size == 1
            self.cabac.encode_bit(1, model=self.unit_flags_model)

            value = sub_block.flatten()[0]
            upper_bitplane = self._get_bitplane(block_position)
            self.encode_value(value, upper_bitplane)

        elif flag == "z":  # Zero
            assert sub_block.size == 1
            self.cabac.encode_bit(0, model=self.unit_flags_model)

        else:
            raise ValueError(f'Invalid encoding flag "{flag}"')

    def encode_value(self, value: int, upper_bitplane: int = 32):
        for i in range(0, upper_bitplane):
            bit = (1 << i) & np.abs(value) != 0
            self.cabac.encode_bit(bit, model=self.bitplane_models[i])
        self.cabac.encode_bit(value < 0, model=self.signals_model)

    def _recursive_optimize_encoding_tree(
        self, block_position: tuple[slice]
    ) -> tuple[str, RD]:
        sub_block = self.block[block_position]

        if sub_block.size == 0:
            return "", RD()

        if sub_block.size == 1:
            return self._estimate_unit_block(block_position)
    
        if np.sum(sub_block) == 0:
            return self._estimate_E_flag(block_position)
    
        if np.all(sub_block != 0):
            return self._estimate_F_flag(block_position)

        best_flags = ""
        best_rd = RD()
        best_cost = float("inf")
        best_models = []
        original_models = self._get_models()

        functions_to_estimate = [
            self._estimate_E_flag,
            self._estimate_F_flag,
            self._estimate_S_flag,
        ]

        for func in functions_to_estimate:
            flags, rd = func(block_position)
            cost = rd.cost(self.lagrangian)
            models = self._get_models()
            self._set_models(original_models)

            if cost < best_cost:
                best_flags = flags
                best_rd = rd
                best_cost = cost
                best_models = models

        self._set_models(best_models)
        return best_flags, best_rd

    def _estimate_E_flag(self, block_position: tuple[slice]) -> tuple[str, RD]:
        sub_block = self.block[block_position]
        self.block_flags_model.add_bit(0)
        self.block_flags_model.add_bit(1)
        return "E", RD(
            self._estimate_current_rate(),
            energy(sub_block),
        )

    def _estimate_F_flag(self, block_position: tuple[slice]) -> tuple[str, RD]:
        self.block_flags_model.add_bit(0)
        self.block_flags_model.add_bit(0)

        sub_block = self.block[block_position]
        dims = np.mgrid[block_position]
        levels = np.max(dims, axis=0)
        for level, value in zip(levels.flatten(), sub_block.flatten()):
            upper_bitplane = self.bitplane_sizes[level]
            for i in range(0, upper_bitplane):
                bit = (1 << i) & np.abs(value) != 0
                self.bitplane_models[i].add_bit(bit)
            self.signals_model.add_bit(value < 0)

        return "F", RD(
            self._estimate_current_rate(),
            0,
        )

    def _estimate_S_flag(self, block_position: tuple[slice]) -> tuple[str, RD]:
        self.block_flags_model.add_bit(1)

        distortion = 0
        flags = "S"
        for sub_pos in split_shape_in_half(block_position):
            current_flags, current_rd = self._recursive_optimize_encoding_tree(sub_pos)
            flags += current_flags
            distortion += current_rd.distortion

        return flags, RD(
            self._estimate_current_rate(),
            distortion,
        )

    def _estimate_unit_block(self, block_position: tuple[slice]) -> tuple[str, RD]:
        value = self.block[block_position].flatten()[0]
        flags = ""

        if value != 0:
            flags = "v"
            self.unit_flags_model.add_bit(1)
            upper_bitplane = self._get_bitplane(block_position)
            for i in range(0, upper_bitplane):
                bit = (1 << i) & np.abs(value) != 0
                self.bitplane_models[i].add_bit(bit)
            self.signals_model.add_bit(value < 0)
        else:
            flags = "z"
            self.unit_flags_model.add_bit(0)

        return flags, RD(
            self._estimate_current_rate(),
            0,
        )

    def _calculate_bitplane_sizes(self):
        tmp_block = self.block.copy()
        bitplane_sizes = []

        for i in range(max(self.block.shape)):
            slices = tuple(slice(0, i) for _ in range(self.block.ndim))
            tmp_block[*slices] = 0

            bp = self.find_max_bitplane(tmp_block)
            bitplane_sizes.append(bp)

        return bitplane_sizes

    def _get_bitplane(self, block_position: tuple[slice]) -> int:
        level = max(s.start for s in block_position)
        return self.bitplane_sizes[level]

    def _get_models(self):
        values = []
        for model in self.probability_models():
            values.append(model.get_values())
        return values

    def _set_models(self, new_values: list):
        for values, model in zip(new_values, self.probability_models()):
            model.set_values(values)

    def _clear_models(self):
        for model in self.probability_models():
            model.clear()

    def _estimate_current_rate(self) -> float:
        total_size = 0
        for model in self.probability_models():
            total_size += model.total_estimated_rate()
        return total_size

    def probability_models(self):
        return [
            self.unit_flags_model,
            self.block_flags_model,
            self.signals_model,
            self.bitplane_sizes_model,
            *self.bitplane_models,
        ]

    @staticmethod
    def find_max_bitplane(block: np.ndarray):
        max_abs = np.max(np.abs(block))
        return int(max_abs).bit_length()
