class VirtualClock:
    def __init__(self):
        self.time = 0.0

    def __call__(self):
        return self.time

    def advance(self, dt):
        self.time += dt


class MockEngine:
    def __init__(self, clock):
        self.clock = clock
        self.prefill_calls = 0
        self.decode_calls = 0
        self.setup_args = None
        self.mem = 100.0
        self.energy = 10.0
        self.p_time = 0.5
        self.d_time = 0.05
        self.d_mem = 0.0
        self.d_energy = 0.0

    def setup(self, context, batch, quant):
        self.setup_args = (context, batch, quant)

    def prefill(self, prompt_tokens):
        self.prefill_calls += 1
        self.clock.advance(self.p_time)
        self.mem += self.d_mem
        self.energy += self.d_energy

    def decode(self, token):
        self.decode_calls += 1
        self.clock.advance(self.d_time)
        self.mem += self.d_mem
        self.energy += self.d_energy

    def memory_usage(self):
        return self.mem

    def energy_usage(self):
        return self.energy
