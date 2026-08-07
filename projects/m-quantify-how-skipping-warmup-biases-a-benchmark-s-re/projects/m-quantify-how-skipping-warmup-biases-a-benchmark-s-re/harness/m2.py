import ref
import numpy as np

def check(workdir):
    from bench.harness import BenchmarkHarness
    from bench.analysis import quantify_bias, explain_mlx_vs_llama
    out = {"bias_quantified": 0.0, "rel_err": 1.0}
    try:
        h = BenchmarkHarness(256, 256, "float16", "mlx")
        lats = h.run(warmup=5, iters=30)
        got_bias = quantify_bias(lats, warmup=5)
        ref_bias = ref.compute_warmup_bias(lats, warmup=5)
        err = abs(got_bias - ref_bias) / (abs(ref_bias) + 1e-8)
        out["bias_quantified"] = 1.0
        out["rel_err"] = float(err)
        h_llama = BenchmarkHarness(256, 256, "float16", "llama_cpp")
        comp = explain_mlx_vs_llama(lats, h_llama.run(warmup=5, iters=30), warmup=5)
        if not isinstance(comp, dict) or "mlx_slower" not in comp:
            out["bias_quantified"] = 0.0
    except Exception as e:
        out["_note"] = str(e)
    return out
