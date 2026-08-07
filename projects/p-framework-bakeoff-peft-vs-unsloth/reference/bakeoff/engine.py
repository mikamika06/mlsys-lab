import numpy as np


class BakeoffEngine:
    def __init__(self, config):
        self.config = config
        self.seed = config.get("seed", 42)
        self.backends = ["baseline", "peft_mock", "optimized"]
        self.weights = {b: np.ones((16, 16), dtype=np.float32) * 0.1 for b in self.backends}
        self.data_state = 0

    def prepare_data(self):
        np.random.seed(self.seed)
        return np.random.randn(100, 16).astype(np.float32)

    def step(self, backend_id):
        rng = np.random.default_rng(self.seed + self.data_state)
        grad = rng.standard_normal((16, 16), dtype=np.float32) * 0.01
        self.weights[backend_id] -= grad
        self.data_state += 1
        step_time = 0.05 + rng.random() * 0.01
        peak_mem = 1024.0 + rng.random() * 50.0
        return {"time": float(step_time), "memory": float(peak_mem)}

    def get_weights(self, backend_id):
        return self.weights[backend_id]

    def evaluate(self, backend_id):
        w = self.weights[backend_id]
        score = float(np.mean(w) * 10.0 + 0.8)
        return score

    def run_benchmark(self, runs=3):
        results = {}
        for b in self.backends:
            times = []
            mems = []
            for _ in range(runs):
                self.data_state = 0
                self.weights[b] = np.ones((16, 16), dtype=np.float32) * 0.1
                for _ in range(5):
                    m = self.step(b)
                    times.append(m["time"])
                    mems.append(m["memory"])
            results[b] = {
                "mean_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
                "mean_memory": float(np.mean(mems)),
                "std_memory": float(np.std(mems)),
                "eval_score": self.evaluate(b)
            }
        return results

    def recommend(self):
        res = self.run_benchmark(runs=3)
        best = min(res, key=lambda x: res[x]["mean_time"])
        return {"recommended": best, "metrics": res[best]}
