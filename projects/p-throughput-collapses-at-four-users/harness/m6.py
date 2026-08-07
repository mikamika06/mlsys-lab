import importlib.util
import os
import sys

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import importlib
    engine_mod = importlib.import_module("runner.engine")
    bench_mod = importlib.import_module("runner.bench")
    qmodel_mod = importlib.import_module("runner.queue_model")

    out = {"prediction_accurate": 0.0, "has_tests": 0.0, "faults_caught": 0.0}

    try:
        cfg = engine_mod.EngineConfig()
        model = qmodel_mod.QueueModel(cfg)
        bench = bench_mod.LoadBench(warmup_runs=0)
        wl8 = bench.generate_workload(num_users=8, prompt_len=32, output_len=50)
        engine = engine_mod.Engine(cfg)
        sim_res = bench.run_benchmark(engine, wl8)
        sim_p95 = sim_res["p95_latency_ms"]

        pred_p95 = model.predict_p95_latency(num_users=8, prompt_len=32, output_len=50)
        rel_error = abs(pred_p95 - sim_p95) / sim_p95 if sim_p95 > 0 else 1.0
        if rel_error <= 0.15:
            out["prediction_accurate"] = 1.0
    except Exception:
        pass

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["_note"] = f"Learner tests fail on good implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0

    good_predict = qmodel_mod.QueueModel.predict_p95_latency
    def naive_predict(self, num_users, prompt_len, output_len):
        t_prefill = prompt_len * self.config.prefill_ms_per_tok
        t_decode = output_len * (self.config.decode_base_ms + num_users * self.config.decode_per_slot_ms)
        return t_prefill + t_decode
    qmodel_mod.QueueModel.predict_p95_latency = naive_predict
    f1 = 0.0 if _survives(path) else 1.0
    qmodel_mod.QueueModel.predict_p95_latency = good_predict

    good_run = engine_mod.Engine.run_trace
    def broken_run(self, requests):
        cfg_copy = engine_mod.EngineConfig(
            gpu_memory_mb=self.config.gpu_memory_mb * 10,
            bytes_per_slot_mb=self.config.bytes_per_slot_mb,
            max_batch_size=self.config.max_batch_size * 10,
            prefill_ms_per_tok=self.config.prefill_ms_per_tok,
            decode_base_ms=self.config.decode_base_ms,
            decode_per_slot_ms=self.config.decode_per_slot_ms
        )
        fake_engine = engine_mod.Engine(cfg_copy)
        return good_run(fake_engine, requests)
    engine_mod.Engine.run_trace = broken_run
    f2 = 0.0 if _survives(path) else 1.0
    engine_mod.Engine.run_trace = good_run

    out["faults_caught"] = f1 + f2
    return out
