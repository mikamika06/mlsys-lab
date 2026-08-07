class Benchmark:
    def __init__(self, clock_fn):
        self.clock = clock_fn

    def run_basic(self, engine, prompt_tokens, gen_len):
        engine.prefill(prompt_tokens)
        for i in range(gen_len):
            engine.decode(i)
        return gen_len

    def setup_engine(self, engine, context, batch, quant):
        engine.setup(context, batch, quant)
        return True

    def run_perf(self, engine, prompt_tokens, gen_len):
        e_start = engine.energy_usage()
        m_start = engine.memory_usage()

        t0 = self.clock()
        engine.prefill(prompt_tokens)
        t1 = self.clock()
        prefill_time = t1 - t0

        t2 = self.clock()
        for i in range(gen_len):
            engine.decode(i)
        t3 = self.clock()
        decode_time_per_token = (t3 - t2) / gen_len if gen_len > 0 else 0.0

        e_end = engine.energy_usage()
        m_end = engine.memory_usage()

        return {
            "prefill_time": prefill_time,
            "decode_time_per_token": decode_time_per_token,
            "memory_peak": m_end - m_start,
            "energy_used": e_end - e_start
        }

    def run_stable(self, engine, prompt_tokens, gen_len, runs=3):
        results = []
        for _ in range(runs):
            results.append(self.run_perf(engine, prompt_tokens, gen_len))

        def median(lst):
            s = sorted(lst)
            return s[len(s) // 2]

        return {
            k: median([r[k] for r in results])
            for k in results[0].keys()
        }
