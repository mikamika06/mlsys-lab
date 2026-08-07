import numpy as np

class ModelBenchmarker:
    def __init__(self, original_model, optimized_model):
        self.original_model = original_model
        self.optimized_model = optimized_model

    def benchmark_phases(self, prefill_shape, decode_shape, runs=10):
        from ref import simulate_execution_time

        orig_prefill = simulate_execution_time(self.original_model, prefill_shape, runs)
        opt_prefill = simulate_execution_time(self.optimized_model, prefill_shape, runs)

        orig_decode = simulate_execution_time(self.original_model, decode_shape, runs)
        opt_decode = simulate_execution_time(self.optimized_model, decode_shape, runs)

        return {
            "prefill": {
                "original_ms": orig_prefill,
                "optimized_ms": opt_prefill,
                "speedup": orig_prefill / opt_prefill if opt_prefill > 0 else 1.0
            },
            "decode": {
                "original_ms": orig_decode,
                "optimized_ms": opt_decode,
                "speedup": orig_decode / opt_decode if opt_decode > 0 else 1.0
            }
        }

    def verify_speedup_and_parity(self, input_shape, threshold=1.15):
        from ref import simulate_execution_time, run_model

        data = np.random.randn(*input_shape).astype(np.float32)
        orig_out = run_model(self.original_model, data)
        opt_out = run_model(self.optimized_model, data)

        parity = bool(np.allclose(orig_out, opt_out, rtol=1e-3, atol=1e-4))

        t_orig = simulate_execution_time(self.original_model, input_shape, runs=5)
        t_opt = simulate_execution_time(self.optimized_model, input_shape, runs=5)

        speedup = t_orig / t_opt if t_opt > 0 else 1.0
        meets_threshold = bool(speedup >= threshold)

        return {
            "parity_preserved": parity,
            "speedup": speedup,
            "threshold": threshold,
            "meets_threshold": meets_threshold
        }

    def validate_constraints(self, seq_len, num_heads, hidden_size):
        valid_head_dim = (hidden_size % num_heads == 0)
        valid_seq_len = (seq_len > 0 and seq_len <= 4096)
        valid_head_size = ((hidden_size // num_heads) in [32, 64, 128]) if valid_head_dim else False

        supported = valid_head_dim and valid_seq_len and valid_head_size

        reasons = []
        if not valid_head_dim:
            reasons.append("hidden_size must be divisible by num_heads")
        if not valid_seq_len:
            reasons.append("seq_len must be between 1 and 4096")
        if not valid_head_size:
            reasons.append("head dimension (hidden_size / num_heads) must be 32, 64, or 128")

        return {
            "supported": supported,
            "reasons": reasons
        }
