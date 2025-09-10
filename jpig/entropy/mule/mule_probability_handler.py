from jpig.entropy.probability_model.frequentist_pm import FrequentistPM


class MuleProbabilityHandler:
    def __init__(self):
        self.signals_probability_model = FrequentistPM()
        self.flag_probability_models = [FrequentistPM() for _ in range(32 * 2)]
        self.integer_probability_models = [FrequentistPM() for _ in range(32)]

    def signal_model(self) -> FrequentistPM:
        return self.signals_probability_model

    def int_model(self, bitplane: int) -> FrequentistPM:
        assert 0 <= bitplane < 32
        return self.integer_probability_models[bitplane]

    def flag_model(self, bitplane: int, position: int) -> FrequentistPM:
        assert position in (0, 1)
        assert 0 <= bitplane < 32
        return self.flag_probability_models[bitplane * 2 + position]

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
            *self.flag_probability_models,
            *self.integer_probability_models,
        )
