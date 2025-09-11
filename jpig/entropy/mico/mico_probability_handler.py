from jpig.entropy.probability_model.frequentist_pm import FrequentistPM


class MicoProbabilityHandler:
    def __init__(self):
        self._bitplane_sizes_model = FrequentistPM()
        self._signals_probability_model = FrequentistPM()
        self._integer_probability_models = [FrequentistPM() for _ in range(32)]

        self._unit_flags_model = FrequentistPM()
        self._split_flags_model = FrequentistPM()
        self._block_flags_model = FrequentistPM()

    def signal_model(self) -> FrequentistPM:
        return self._signals_probability_model

    def int_model(self, bitplane: int) -> FrequentistPM:
        assert 0 <= bitplane < 32
        return self._integer_probability_models[bitplane]

    def split_model(self) -> FrequentistPM:
        return self._split_flags_model

    def block_model(self) -> FrequentistPM:
        return self._block_flags_model

    def unit_model(self) -> FrequentistPM:
        return self._unit_flags_model

    def bitplanes_model(self) -> FrequentistPM:
        return self._bitplane_sizes_model

    def push(self):
        for model in self._all_models():
            model.push()

    def pop(self):
        for model in self._all_models():
            model.pop()

    def clear(self):
        for model in self._all_models():
            model.clear()

    def estimate_rate(self) -> float:
        rate = 0
        for model in self._all_models():
            rate += model.total_estimated_rate()
        return rate

    def copy(self):
        other = MicoProbabilityHandler()
        for model, other_model in zip(self._all_models(), other._all_models()):
            other_model.set_values(model.get_values())
        return other

    def _all_models(self) -> tuple[FrequentistPM, ...]:
        return (
            self._bitplane_sizes_model,
            self._signals_probability_model,
            *self._integer_probability_models,
            self._unit_flags_model,
            self._split_flags_model,
            self._block_flags_model,
        )
