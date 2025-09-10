from jpig.entropy.probability_model.frequentist_pm import FrequentistPM


class MicoProbabilityHandler:
    def __init__(self):
        self.signals_probability_model = FrequentistPM()
        self.integer_probability_models = [FrequentistPM() for _ in range(32)]

        self.unit_flags_model = FrequentistPM()
        self.split_flags_model = FrequentistPM()
        self.block_flags_model = FrequentistPM()

    def signal_model(self) -> FrequentistPM:
        return self.signals_probability_model

    def int_model(self, bitplane: int) -> FrequentistPM:
        assert 0 <= bitplane < 32
        return self.integer_probability_models[bitplane]

    def split_model(self) -> FrequentistPM:
        return self.split_flags_model

    def block_model(self) -> FrequentistPM:
        return self.block_flags_model

    def unit_model(self) -> FrequentistPM:
        return self.unit_flags_model

    def push(self):
        for model in self._all_models():
            model.push()

    def pop(self):
        for model in self._all_models():
            model.pop()

    def clear(self):
        for model in self._all_models():
            model.clear()

    def _all_models(self):
        return (
            self.signals_probability_model,
            *self.integer_probability_models,
            self.unit_flags_model,
            self.split_flags_model,
            self.block_flags_model,
        )
